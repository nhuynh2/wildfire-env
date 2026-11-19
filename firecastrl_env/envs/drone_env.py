"""DroneWildfireEnv: WildfireEnv adapted to a drone with a thermal IR sensor configuration (partial observation).

Unchanged from the framework: Discrete(5) actions (0:+y 1:-y 2:-x 3:+x 4:drop), the fire engine and
its dynamics, and the step/reset flow.

Added: rescaled environment configuration, one full-load retardant bomb (payload 1 -> 0), a 3x3-cell drop
footprint, a battery that move cost drains, and a thermal sensor square
(DroneConfig.sensor_radius) that is the agent's only view of the fire.

Observation (partial):
  local    : float32 (4, 2R+1, 2R+1) thermal view centered on the drone, rows = y, cols = x, channels
             [burning, burnt, nonburnable, out_of_bounds]; the view is the full square sensor footprint.
  position : float32 (2,) [x/(W-1), y/(H-1)]
  battery  : float32 (1,) remaining battery fraction (1.0 = full)
  time     : float32 (1,) step/MAX_TIMESTEPS (lets the agent anticipate truncation)

Drop interlock: the payload is released only if a burning cell is inside the sensor view. Otherwise action 4
is a hover (no release, no termination). A release always ends the episode.
"""
from __future__ import annotations

import random
from typing import Any, Optional

import numpy as np
from gymnasium import spaces

from . import config
from .drone_config import DroneConfig
from .drone_reward import RewardCoeffs, drone_reward
from .environment.enums import FireState
from .wildfire_env import WildfireEnv

LOCAL_CHANNELS = ("burning", "burnt", "nonburnable", "out_of_bounds")


class DroneWildfireEnv(WildfireEnv):
    def __init__(
        self,
        drone_cfg: Optional[DroneConfig] = None,
        reward_coeffs: Optional[RewardCoeffs] = None,
        battery_enabled: bool = True,
        random_spark: bool = False,
        min_spark_dist_cells: int = 30,
        max_spark_dist_cells: Optional[int] = None,
        **kwargs,
    ):
        self.dcfg = drone_cfg or DroneConfig()
        self.rcoef = reward_coeffs or RewardCoeffs()
        self.battery_enabled = battery_enabled
        self.random_spark = random_spark
        self.min_spark_dist_cells = min_spark_dist_cells
        self.max_spark_dist_cells = max_spark_dist_cells
        self.dcfg.apply_scale(config)  # mutate framework config BEFORE the base reads it

        # sensor square offsets (fixed for the episode)
        offs = np.array(self.dcfg.sensor_offsets(), dtype=np.int32)
        self._sdx, self._sdy = offs[:, 0], offs[:, 1]

        super().__init__(**kwargs)

        side = 2 * self.dcfg.sensor_radius + 1
        self.observation_space = spaces.Dict(
            {
                "local": spaces.Box(0.0, 1.0, shape=(len(LOCAL_CHANNELS), side, side), dtype=np.float32),
                "position": spaces.Box(0.0, 1.0, shape=(2,), dtype=np.float32),
                "battery": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
                "time": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            }
        )

    # --------------------------------- state ---------------------------------
    def _reset_state_variables(self) -> None:
        super()._reset_state_variables()
        self.state["helicopter_coord"] = np.array(self.dcfg.base_coord, dtype=np.int32)
        self.battery = 1.0
        self.payload = self.dcfg.payload_drops
        self._crashed = False
        self._dropped = False
        self._released_now = False
        self._interlocked_now = False
        self._fire_in_view = False
        self._reward_terms: dict[str, float] = {}
        self._reset_episode_tracking()

    def _reset_episode_tracking(self) -> None:
        self._first_detect_step: Optional[int] = None
        self._release_step: Optional[int] = None
        self._ep = {"quenched": 0, "released": 0, "interlocked": 0}

    # --------------------------------- sensor ---------------------------------
    def _sense(self, x: int, y: int) -> np.ndarray:
        """Thermal view: true fire state of the cells inside the square sensor footprint centered on (x, y)."""
        s = self.cell_state
        r = self.dcfg.sensor_radius
        xs, ys = x + self._sdx, y + self._sdy
        inb = (xs >= 0) & (xs < s.width) & (ys >= 0) & (ys < s.height)
        view = np.zeros((len(LOCAL_CHANNELS), 2 * r + 1, 2 * r + 1), dtype=np.float32)

        lx, ly = self._sdx[inb] + r, self._sdy[inb] + r
        cx, cy = xs[inb], ys[inb]
        fs = s.fire_state[cy, cx]
        view[0, ly, lx] = fs == FireState.Burning
        view[1, ly, lx] = fs == FireState.Burnt
        view[2, ly, lx] = s.is_nonburnable[cy, cx]
        view[3, self._sdy[~inb] + r, self._sdx[~inb] + r] = 1.0
        return view

    def _burning_in_view(self, x: int, y: int) -> bool:
        s = self.cell_state
        xs, ys = x + self._sdx, y + self._sdy
        inb = (xs >= 0) & (xs < s.width) & (ys >= 0) & (ys < s.height)
        return bool(np.any(s.fire_state[ys[inb], xs[inb]] == FireState.Burning))

    # --------------------------------- action ---------------------------------
    def _drop(self, x: int, y: int) -> int:
        """Same effects as helper.perform_helitack_array, but over the drone's footprint."""
        s = self.cell_state
        quenched = 0
        for dx, dy in self.dcfg.footprint():
            cx, cy = x + dx, y + dy
            if 0 <= cx < s.width and 0 <= cy < s.height:
                s.helitack_drops[cy, cx] += 1
                s.ignition_time[cy, cx] = np.inf
                if s.fire_state[cy, cx] == FireState.Burning:
                    s.fire_state[cy, cx] = FireState.Unburnt
                    quenched += 1
        self.helitack_history.append((x, y, self.step_count))
        self.helitack_history = [(a, b, c) for a, b, c in self.helitack_history if self.step_count - c < 20]
        return quenched

    def apply_action(self, action: int) -> tuple[int, int, int]:
        d = self.dcfg
        loaded = self.payload > 0
        quenched = 0
        self._released_now = False
        self._interlocked_now = False
        cost = 0.0
        if action == 4:
            hx, hy, _ = super().apply_action(-1)  # no move; base increments step_count
            self.state["last_action"] = 4
            if loaded and self._burning_in_view(hx, hy):
                quenched = self._drop(hx, hy)
                self._released_now = True
                self._dropped = True
                self.payload -= 1
                self._release_step = self.step_count
                self._ep["released"] += 1
            else:
                # interlock (no thermal target in view) or empty payload: the drone just hovers
                self._interlocked_now = loaded
                self._ep["interlocked"] += int(loaded)
        else:
            hx, hy, quenched = super().apply_action(action)
            cost = d.move_cost(loaded)

        self._ep["quenched"] += quenched

        if self.battery_enabled:
            self.battery = max(0.0, self.battery - cost)

        if self.battery_enabled and self.battery <= 0.0:
            self._crashed = True
        return hx, hy, quenched

    # --------------------------------- reward ---------------------------------
    def calculate_reward(self, curr_burning: int, extinguished_by_helitack: int) -> float:
        reward, self._reward_terms = drone_reward(
            self.rcoef,
            quenched=extinguished_by_helitack,
            released=self._released_now,
            crashed=self._crashed,
            battery=self.battery,
        )
        return reward

    # --------------------------------- observation ---------------------------------
    def _build_observation(self, quenched_cells: int) -> dict[str, np.ndarray]:
        hx, hy = (int(v) for v in self.state["helicopter_coord"])
        local = self._sense(hx, hy)
        self._fire_in_view = bool(local[0].any())
        if self._fire_in_view and self._first_detect_step is None:
            self._first_detect_step = self.step_count
        return {
            "local": local,
            "position": np.array([hx / max(1, self.gridWidth - 1), hy / max(1, self.gridHeight - 1)], dtype=np.float32),
            "battery": np.array([self.battery], dtype=np.float32),
            "time": np.array([min(1.0, self.step_count / config.MAX_TIMESTEPS)], dtype=np.float32),
        }

    # --------------------------------- reset/step ---------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None):
        if seed is not None:
            random.seed(seed)  # the fire engine uses Python's random, which the base does not seed
        _, info = super().reset(seed=seed, options=options)

        if self.random_spark:
            self._move_spark()
            self._publish_viewer_state()
        self._reset_episode_tracking()
        sx, sy = self._spark_grid_coord
        bx, by = self.dcfg.base_coord
        info["spark_coord"] = (sx, sy)
        info["spark_dist_cells"] = float(np.hypot(sx - bx, sy - by))
        return self._build_observation(0), info

    def _publish_viewer_state(self, *, terminated: bool = False, truncated: bool = False, publish_terrain: bool = False) -> None:
        terminated = terminated or self._crashed or self._dropped
        super()._publish_viewer_state(terminated=terminated, truncated=truncated, publish_terrain=publish_terrain)

    def _move_spark(self) -> None:
        """Randomise the ignition cell (initial condition only; fire dynamics untouched)."""
        s = self.cell_state
        # the base reset() always lights the center cell; _spark_grid_coord may hold last episode's spark
        cx = int((config.modelWidth / 2 - 1) // self.cell_size)
        cy = int((config.modelHeight / 2 - 1) // self.cell_size)
        s.ignition_time[cy, cx] = np.inf
        bx, by = self.dcfg.base_coord
        ys, xs = np.nonzero(~s.is_nonburnable)
        dist = np.hypot(xs - bx, ys - by)
        ok = dist >= self.min_spark_dist_cells
        if self.max_spark_dist_cells is not None:
            ok &= dist <= self.max_spark_dist_cells
        margin = 3
        ok &= (xs >= margin) & (xs < s.width - margin) & (ys >= margin) & (ys < s.height - margin)
        xs, ys = xs[ok], ys[ok]
        if len(xs) == 0:
            raise ValueError(
                f"no burnable spark cell with {self.min_spark_dist_cells} <= dist <= {self.max_spark_dist_cells}"
            )
        i = int(self.np_random.integers(len(xs)))
        self._spark_grid_coord = (int(xs[i]), int(ys[i]))
        s.ignition_time[self._spark_grid_coord[1], self._spark_grid_coord[0]] = 0.0

    def step(self, action: int):
        obs, reward, terminated, truncated, info = super().step(action)
        terminated = bool(terminated or self._crashed or self._dropped)
        info.update(
            battery=self.battery,
            payload=self.payload,
            released=self._released_now,
            interlocked=self._interlocked_now,
            fire_in_view=self._fire_in_view,
            crashed=self._crashed,
            reward_terms=dict(self._reward_terms),
        )
        if terminated or truncated:
            burning, burnt = self._get_current_fire_stats()
            total = self._ep["quenched"] + burnt
            released = self._ep["released"] > 0
            info["episode_metrics"] = {
                **self._ep,
                "success": released and self._ep["quenched"] > 0,  # release landed on >= 1 burning cell
                "wasted_release": released and self._ep["quenched"] == 0,
                "detected": self._first_detect_step is not None,
                "steps_to_detection": self._first_detect_step,
                "steps_to_release": self._release_step,
                "crashed": self._crashed,
                "cells_burnt": burnt,
                "cells_burning": burning,
                "suppression_efficiency": self._ep["quenched"] / total if total else 0.0,
                "steps": self.step_count,
            }
        return obs, reward, terminated, truncated, info
