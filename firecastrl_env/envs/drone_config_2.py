"""Drone scope v2: the framework's scale, unchanged, plus a drone-specific sensor and drop.

Unlike drone_config.DroneConfig, nothing is rescaled to physical drone units. The agent acts on cells,
so the grid, cell size, fire clock, burn times, terrain elevation, and helicopter speed stay at the
framework defaults (envs/config.py). The drone is 2D on the grid with no altitude, like the
framework helicopter.

Added on top of the framework:
  thermal sensor : square (2r+1)x(2r+1) view centered on the drone (agent's only view of the fire)
  drop footprint : 13-cell disk, radius 2 cells (full disk; the framework's helper drops the west tip)
  payload        : one drop per episode

Usage:
    env = DroneWildfireEnv(drone_cfg=DroneConfig2())
"""
from __future__ import annotations

import importlib.util
import math
from dataclasses import dataclass
from typing import Optional

# Framework config keys that drone_config.DroneConfig.apply_scale overwrites. apply_scale below resets them
# to the values in envs/config.py, so a v1 env built earlier in the same process can't leak its scale into v2.
_SCALE_KEYS = (
    "gridWidth", "gridHeight", "cellSize", "modelWidth", "modelHeight", "heightmapMaxElevation",
    "minCellBurnTime", "helitackDropRadius", "HELICOPTER_SPEED", "MAX_TIMESTEPS", "maxTimeStep",
)


def _framework_defaults(config) -> dict:
    """Read the default values of _SCALE_KEYS from envs/config.py by loading a fresh copy of that file."""
    spec = importlib.util.spec_from_file_location(f"{config.__package__}._config_defaults", config.__file__)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load framework config from {config.__file__}")
    fresh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fresh)
    return {k: getattr(fresh, k) for k in _SCALE_KEYS}


@dataclass
class DroneConfig2:
    # --- drop ---
    drop_radius_cells: float = 2.0     # 13-cell disk
    payload_drops: int = 1             # one drop per episode

    # --- thermal IR sensor (square, north-up) ---
    # Half-side r in cells: the view is (2r+1)x(2r+1) centered on the drone.
    sensor_radius: int = 20

    # --- mission ---
    # Base sits base_offset_cells west of the framework's map-center spark.
    base_offset_cells: int = 75
    base_coord: Optional[tuple] = None

    def __post_init__(self):
        if self.base_coord is None:
            # Matches how wildfire_env.reset() picks the spark cell: (119, 79) on the default grid.
            from . import config
            spark_x = int((config.modelWidth / 2 - 1) // config.cellSize)
            spark_y = int((config.modelHeight / 2 - 1) // config.cellSize)
            self.base_coord = (spark_x - self.base_offset_cells, spark_y)

    # ---------- drop ----------
    def footprint(self) -> list[tuple[int, int]]:
        """(dx, dy) offsets of cells whose centers lie within drop_radius_cells of the drone."""
        r = self.drop_radius_cells
        n = math.floor(r)
        return [
            (dx, dy)
            for dy in range(-n, n + 1)
            for dx in range(-n, n + 1)
            if dx * dx + dy * dy <= r * r
        ]

    # ---------- thermal sensor ----------
    def sensor_offsets(self) -> list[tuple[int, int]]:
        """(dx, dy) offsets of the cells inside the square sensor view (axis-aligned, north-up)."""
        r = self.sensor_radius
        return [(dx, dy) for dy in range(-r, r + 1) for dx in range(-r, r + 1)]

    # ---------- plumbing into the framework ----------
    def apply_scale(self, config) -> None:
        """Keep the framework's scale. Apply the drone's drop area."""
        for key, value in _framework_defaults(config).items():
            setattr(config, key, value)
        config.helitackDropRadius = self.drop_radius_cells * config.cellSize

    def summary(self) -> str:
        from . import config
        step_min = min(config.maxTimeStep, 86400 / config.modelDayInSeconds * 0.000277 * 4)  # as in WildfireEnv
        cell = config.cellSize
        speed = config.HELICOPTER_SPEED
        side = 2 * self.sensor_radius + 1
        n_view = len(self.sensor_offsets())
        n_map = config.gridWidth * config.gridHeight
        base_steps = math.ceil(self.base_offset_cells / speed)
        return "\n".join([
            "DroneConfig2 (framework scale, nothing rescaled)",
            f"  Map     : {config.gridWidth}x{config.gridHeight} cells",
            f"  Clock   : {step_min:.2f} min of fire per step; {config.MAX_TIMESTEPS} steps"
            f" = {config.MAX_TIMESTEPS * step_min / 1440:.1f} days per episode",
            f"  Drone   : moves {speed} cells per step, 2D on the grid, no altitude",
            f"  Base    : at {self.base_coord}, {self.base_offset_cells} cells"
            f" west of the spark, ~{base_steps} steps away",
            f"  Drop    : {len(self.footprint())}-cell disk, radius {self.drop_radius_cells:g} cells;"
            f" {self.payload_drops} per episode",
            f"  Sensor  : {side}x{side} cells centered on drone,"
            f" {n_view:,} cells = {100 * n_view / n_map:.1f}% of map",
        ])


if __name__ == "__main__":
    from . import config
    cfg = DroneConfig2()
    cfg.apply_scale(config)
    print(cfg.summary())
