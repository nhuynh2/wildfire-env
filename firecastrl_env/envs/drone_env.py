"""DroneWildfireEnv: WildfireEnv adapted for two-stage MaskablePPO training and full-mission eval.

Stages:
  stage="detect" : Task 1 (Search & Detect) - zeros channels 1..3, masks action 4, ends on detection or max_steps.
  stage="drop"   : Task 2 (Target & Drop)   - all 4 channels, all 5 actions, pre-ticks fire & spawns near it.
  stage="eval"   : Full end-to-end mission  - default base_coord, all 4 channels, all 5 actions, base reward.
"""
from __future__ import annotations

import math
import random
from typing import Any, Optional

import numpy as np
from gymnasium import spaces

from . import config
from .drone_config_2 import DroneConfig2 as DroneConfig
from .drone_reward import RewardCoeffs, detect_reward, drone_reward, drop_reward
from .environment.enums import FireState
from .wildfire_env import WildfireEnv

LOCAL_CHANNELS = ("burning", "burnt", "nonburnable", "out_of_bounds")

class DroneWildfireEnv(WildfireEnv):
    def __init__(
        self,
        drone_cfg: Optional[DroneConfig] = None,
        reward_coeffs: Optional[RewardCoeffs] = None,
        *,
        stage: str = "eval",
        max_steps: Optional[int] = None,
        randomize_spawn: bool = True,
        min_spawn_dist: float = 55.0,
        max_spawn_dist: float = 75.0,
        min_pre_ticks: int = 10,
        max_pre_ticks: int = 50,
        **kwargs,
    ):
        assert stage in ("detect", "drop", "eval"), f"Unknown stage: {stage}"
        self.stage = stage
        self.max_steps = max_steps if max_steps is not None else (150 if stage == "detect" else 50)
        self.randomize_spawn = randomize_spawn
        self.min_spawn_dist = min_spawn_dist
        self.max_spawn_dist = max_spawn_dist
        self.min_pre_ticks = min_pre_ticks
        self.max_pre_ticks = max_pre_ticks

        self.dcfg = drone_cfg or DroneConfig()
        self.rcoef = reward_coeffs or RewardCoeffs()
        self.dcfg.apply_scale(config)

        offs = np.array(self.dcfg.sensor_offsets(), dtype=np.int32)
        self._sdx, self._sdy = offs[:, 0], offs[:, 1]

        super().__init__(**kwargs)

        side = 2 * self.dcfg.sensor_radius + 1
        self.observation_space = spaces.Dict(
            {
                "local": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(len(LOCAL_CHANNELS), side, side),
                    dtype=np.float32,
                ),
                "position": spaces.Box(0.0, 1.0, shape=(2,), dtype=np.float32),
            }
        )

    # --------------------------------- masking ---------------------------------
    def action_masks(self) -> np.ndarray:
        """Used automatically by MaskablePPO (compatible with SubprocVecEnv)."""
        if self.stage == "detect":
            return np.array([True, True, True, True, False], dtype=bool)
        return np.array([True, True, True, True, True], dtype=bool)

    # --------------------------------- state ---------------------------------
    def _reset_state_variables(self) -> None:
        super()._reset_state_variables()
        self.state["helicopter_coord"] = np.array(self.dcfg.base_coord, dtype=np.int32)
        self.payload = self.dcfg.payload_drops
        self._released_now = False
        self._interlocked_now = False
        self._fire_in_view = False
        self._reward_terms: dict[str, float] = {}
        self._seen: Optional[np.ndarray] = None
        self._prev_dist: Optional[float] = None
        self._reset_episode_tracking()

    def _reset_episode_tracking(self) -> None:
        self._first_detect_step: Optional[int] = None
        self._release_step: Optional[int] = None
        self._ep = {"quenched": 0, "released": 0, "interlocked": 0}

    # --------------------------------- sensor & helpers ---------------------------------
    def _sense(self, x: int, y: int) -> np.ndarray:
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

    def _dist_to_nearest_burning(self, x: int, y: int) -> Optional[float]:
        s = self.cell_state
        xs, ys = x + self._sdx, y + self._sdy
        inb = (xs >= 0) & (xs < s.width) & (ys >= 0) & (ys < s.height)
        burning = s.fire_state[ys[inb], xs[inb]] == FireState.Burning
        if not np.any(burning):
            return None
        dists = np.hypot(self._sdx[inb][burning], self._sdy[inb][burning])
        return float(dists.min())

    def _mark_seen(self) -> int:
        if self._seen is None:
            return 0
        hx, hy = (int(v) for v in self.state["helicopter_coord"])
        r = self.dcfg.sensor_radius
        x0, x1 = max(0, hx - r), min(self.gridWidth, hx + r + 1)
        y0, y1 = max(0, hy - r), min(self.gridHeight, hy + r + 1)
        newly_seen = int(np.count_nonzero(~self._seen[y0:y1, x0:x1]))
        self._seen[y0:y1, x0:x1] = True
        return newly_seen

    # --------------------------------- action ---------------------------------
    def _drop(self, x: int, y: int) -> int:
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
        loaded = self.payload > 0
        quenched = 0
        self._released_now = False
        self._interlocked_now = False
        if action == 4:
            hx, hy, _ = super().apply_action(-1)
            self.state["last_action"] = 4
            if loaded and self._burning_in_view(hx, hy):
                quenched = self._drop(hx, hy)
                self._released_now = True
                self.payload -= 1
                self._release_step = self.step_count
                self._ep["released"] += 1
            else:
                self._interlocked_now = loaded
                self._ep["interlocked"] += int(loaded)
        else:
            hx, hy, _ = super().apply_action(action)

        self._ep["quenched"] += quenched
        return hx, hy, quenched

    # --------------------------------- reward ---------------------------------
    def calculate_reward(self, curr_burning: int, extinguished_by_helitack: int) -> float:
        hx, hy = (int(v) for v in self.state["helicopter_coord"])
        if self.stage == "detect":
            newly_seen = self._mark_seen()
            detected = self._burning_in_view(hx, hy)
            reward, self._reward_terms = detect_reward(self.rcoef, detected=detected, newly_seen=newly_seen)
        elif self.stage == "drop":
            curr_dist = self._dist_to_nearest_burning(hx, hy)
            dist_delta = (self._prev_dist - curr_dist) if (curr_dist is not None and self._prev_dist is not None) else 0.0
            self._prev_dist = curr_dist
            reward, self._reward_terms = drop_reward(
                self.rcoef,
                quenched=extinguished_by_helitack,
                released=self._released_now,
                dist_delta=dist_delta,
            )
        else:
            reward, self._reward_terms = drone_reward(
                self.rcoef,
                quenched=extinguished_by_helitack,
                released=self._released_now,
            )
        return reward

    # --------------------------------- observation ---------------------------------
    def _build_observation(self, quenched_cells: int) -> dict[str, np.ndarray]:
        hx, hy = (int(v) for v in self.state["helicopter_coord"])
        local = self._sense(hx, hy)
        self._fire_in_view = bool(local[0].any())
        if self._fire_in_view and self._first_detect_step is None:
            self._first_detect_step = self.step_count
        if self.stage == "detect":
            local[1:] = 0.0  # Task 1 sees burning channel only
        return {
            "local": local,
            "position": np.array(
                [hx / max(1, self.gridWidth - 1), hy / max(1, self.gridHeight - 1)],
                dtype=np.float32,
            ),
        }

    # --------------------------------- reset / step ---------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None):
        if seed is not None:
            random.seed(seed)
        self._reset_episode_tracking()
        obs, info = super().reset(seed=seed, options=options)

        opts = options or {}
        if self.stage == "detect":
            if "drone_coord" in opts:
                dx, dy = opts["drone_coord"]
                self.state["helicopter_coord"] = np.array(
                    [np.clip(int(dx), 0, self.gridWidth - 1), np.clip(int(dy), 0, self.gridHeight - 1)],
                    dtype=np.int32,
                )
            elif self.randomize_spawn:
                sx, sy = self._spark_grid_coord
                r = self.dcfg.sensor_radius
                for _ in range(50):
                    angle = random.uniform(0.0, 2.0 * math.pi)                              # random direction
                    dist = random.uniform(self.min_spawn_dist, self.max_spawn_dist)         # random distance from the fire
                    dx = int(np.clip(sx + dist * math.cos(angle), 1, self.gridWidth - 2))   # step that far in that direction,
                    dy = int(np.clip(sy + dist * math.sin(angle), 1, self.gridHeight - 2))  # clipped to the grid
                    if abs(dx - sx) > r or abs(dy - sy) > r:
                        self.state["helicopter_coord"] = np.array([dx, dy], dtype=np.int32)
                        break
            self._seen = np.zeros((self.gridHeight, self.gridWidth), dtype=bool)
            self._mark_seen()
            obs = self._build_observation(0)

        elif self.stage == "drop":
            pre_ticks = int(opts.get("pre_ticks", random.randint(self.min_pre_ticks, self.max_pre_ticks)))
            for _ in range(pre_ticks):
                self.tick(self._step_delta)
            if "base_coord" in opts:
                dx, dy = opts["base_coord"]
                self.state["helicopter_coord"] = np.array(
                    [np.clip(int(dx), 0, self.gridWidth - 1), np.clip(int(dy), 0, self.gridHeight - 1)],
                    dtype=np.int32,
                )
            else:
                by, bx = np.nonzero(self.cell_state.fire_state == FireState.Burning)
                if len(bx) > 0:
                    idx = random.randrange(len(bx))
                    tx, ty = int(bx[idx]), int(by[idx])
                    r = self.dcfg.sensor_radius
                    hx = int(np.clip(tx + random.randint(-r + 2, r - 2), 0, self.gridWidth - 1))
                    hy = int(np.clip(ty + random.randint(-r + 2, r - 2), 0, self.gridHeight - 1))
                    self.state["helicopter_coord"] = np.array([hx, hy], dtype=np.int32)
            hx, hy = (int(v) for v in self.state["helicopter_coord"])
            self._prev_dist = self._dist_to_nearest_burning(hx, hy)
            obs = self._build_observation(0)

        else:  # stage == "eval"
            pre_ticks = int(opts.get("pre_ticks", 0))
            for _ in range(pre_ticks):
                self.tick(self._step_delta)
            if "base_coord" in opts:
                dx, dy = opts["base_coord"]
                self.state["helicopter_coord"] = np.array(
                    [np.clip(int(dx), 0, self.gridWidth - 1), np.clip(int(dy), 0, self.gridHeight - 1)],
                    dtype=np.int32,
                )
            if pre_ticks > 0 or "base_coord" in opts:
                obs = self._build_observation(0)

        return obs, info

    def _publish_viewer_state(
        self,
        *,
        terminated: bool = False,
        truncated: bool = False,
        publish_terrain: bool = False,
    ) -> None:
        terminated = terminated or self._released_now
        super()._publish_viewer_state(terminated=terminated, truncated=truncated, publish_terrain=publish_terrain)

    def step(self, action: int):
        obs, reward, terminated, truncated, info = super().step(action)
        if self.stage == "detect":
            if self._fire_in_view:
                terminated = True
            elif self.step_count >= self.max_steps:
                truncated = True
        elif self.stage == "drop":
            if self._released_now:
                terminated = True
            elif self.step_count >= self.max_steps and not terminated:
                truncated = True
        else:
            terminated = bool(terminated or self._released_now)

        info.update(
            payload=self.payload,
            released=self._released_now,
            interlocked=self._interlocked_now,
            fire_in_view=self._fire_in_view,
            reward_terms=dict(self._reward_terms),
        )
        if terminated or truncated:
            burning, burnt = self._get_current_fire_stats()
            drop_capacity = len(self.dcfg.footprint()) * self.dcfg.payload_drops  # 13 cells per retardant drop
            released = self._ep["released"] > 0
            info["episode_metrics"] = {
                **self._ep,
                "is_success": released and self._ep["quenched"] > 0,
                "wasted_release": released and self._ep["quenched"] == 0,
                "detected": self._first_detect_step is not None,
                "steps_to_detection": self._first_detect_step,
                "steps_to_release": self._release_step,
                "cells_burnt": burnt,
                "cells_burning": burning,
                "suppression_efficiency": self._ep["quenched"] / drop_capacity,
                "steps": self.step_count,
            }
        return obs, reward, terminated, truncated, info