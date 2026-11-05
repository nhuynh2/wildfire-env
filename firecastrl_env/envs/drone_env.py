"""DroneWildfireEnv: WildfireEnv adapted to a drone (Stage 1: full-map observation).

Unchanged from the framework: Discrete(5) actions (0:+y 1:-y 2:-x 3:+x 4:drop), the fire engine and
its dynamics, the base observation keys, and the step/reset flow.

Added: rescaled environment configuration, one full-load retardant bomb (payload 1 -> 0),
a 3x3-cell drop footprint, refill/recharge near base, a battery that move/drop/hover costs drain.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import random
from gymnasium import spaces

from . import config
from .drone_config import DroneConfig
from .drone_reward import RewardCoeffs, drone_reward
from .environment.enums import FireState
from .wildfire_env import WildfireEnv


class DroneWildfireEnv(WildfireEnv):
    def __init__(
        self,
        drone_cfg: Optional[DroneConfig] = None,
        reward_coeffs: Optional[RewardCoeffs] = None,
        battery_enabled: bool = True,
        random_spark: bool = False,
        min_spark_dist_cells: int = 30,
        **kwargs,
    ):
        self.dcfg = drone_cfg or DroneConfig()
        self.rcoef = reward_coeffs or RewardCoeffs()
        self.battery_enabled = battery_enabled
        self.random_spark = random_spark
        self.min_spark_dist_cells = min_spark_dist_cells
        self.dcfg.apply_scale(config)  # mutate framework config BEFORE the base reads it

        super().__init__(**kwargs)

        obs_spaces = dict(self.observation_space.spaces)
        obs_spaces["battery"] = spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)
        obs_spaces["payload"] = spaces.Box(0.0, float(self.dcfg.payload_drops), shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Dict(obs_spaces)

    # --------------------------------- state --------------------------------- 
    def _reset_state_variables(self) -> None:
        super()._reset_state_variables()
        self.state["helicopter_coord"] = np.array(self.dcfg.base_coord, dtype=np.int32)
        self.battery = 1.0
        self.payload = self.dcfg.payload_drops
        self._wasted = False
        self._crashed = False
        self._dropped = False
        self._reward_terms: dict[str, float] = {}
        self._ep = {"quenched": 0, "drops": 0, "wasted": 0, "refills": 0}

    def _is_near_base(self, x: int, y: int) -> bool:
        bx, by = self.dcfg.base_coord
        return max(abs(x - bx), abs(y - by)) <= self.dcfg.refill_radius

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

        if action == 4:
            hx, hy, _ = super().apply_action(-1)
            self.state["last_action"] = 4
            if loaded:
                quenched = self._drop(hx, hy)
                self._dropped = True  
                self.payload -= 1
                self._ep["drops"] += 1
                cost = d.drop_cost()
            else:
                cost = d.hover_cost(False)  # wasted drop with an empty payload: just hovers
        else:
            hx, hy, quenched = super().apply_action(action)
            cost = d.move_cost(loaded)

        self._wasted = action == 4 and quenched == 0
        self._ep["quenched"] += quenched
        self._ep["wasted"] += int(self._wasted)

        if self.battery_enabled:
            self.battery = max(0.0, self.battery - cost)

        self._near_base = self._is_near_base(hx, hy)
        if self._near_base:
            if self.payload < d.payload_drops:
                self._ep["refills"] += 1
            self.payload = d.payload_drops
            if self.battery_enabled:
                self.battery = 1.0
        elif self.battery_enabled and self.battery <= 0.0:
            self._crashed = True
        return hx, hy, quenched

    # --------------------------------- reward ---------------------------------
    def calculate_reward(self, curr_burning: int, extinguished_by_helitack: int) -> float:
        _, burnt = self._get_current_fire_stats()
        reward, self._reward_terms = drone_reward(
            self.rcoef,
            quenched=extinguished_by_helitack,
            last_action=self.state["last_action"],
            battery=self.battery,
            burning=curr_burning,
        )
        return reward

    # --------------------------------- observation ---------------------------------
    def _build_observation(self, quenched_cells: int) -> dict[str, np.ndarray]:
        obs = super()._build_observation(quenched_cells)
        obs["battery"] = np.array([self.battery], dtype=np.float32)
        obs["payload"] = np.array([self.payload], dtype=np.float32)
        return obs

    # --------------------------------- reset/step ---------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None):
        if seed is not None:
            random.seed(seed)  # the fire engine uses Python's random, which the base does not seed
        _, info = super().reset(seed=seed, options=options)

        if self.random_spark:
            self._move_spark()
            self._publish_viewer_state()
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
        far = np.hypot(xs - bx, ys - by) >= self.min_spark_dist_cells
        margin = 3
        ok = far & (xs >= margin) & (xs < s.width - margin) & (ys >= margin) & (ys < s.height - margin)
        xs, ys = xs[ok], ys[ok]
        i = int(self.np_random.integers(len(xs)))
        self._spark_grid_coord = (int(xs[i]), int(ys[i]))
        s.ignition_time[self._spark_grid_coord[1], self._spark_grid_coord[0]] = 0.0

    def step(self, action: int):
        obs, reward, terminated, truncated, info = super().step(action)
        terminated = bool(terminated or self._crashed or self._dropped)
        info.update(
            battery=self.battery,
            payload=self.payload,
            wasted_drop=self._wasted,
            crashed=self._crashed,
            near_base=self._near_base,
            reward_terms=dict(self._reward_terms),
        )
        if terminated or truncated:
            burning, burnt = self._get_current_fire_stats()
            total = self._ep["quenched"] + burnt
            info["episode_metrics"] = {
                **self._ep,
                "cells_burnt": burnt,
                "cells_burning": burning,
                "suppression_efficiency": self._ep["quenched"] / total if total else 0.0,
                "crashed": self._crashed,
                "steps": self.step_count,
            }
        return obs, reward, terminated, truncated, info
