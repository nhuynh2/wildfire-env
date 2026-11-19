"""Drone scope for DroneWildfireEnv (Spider-i H300 class, ONE full-load bomb, 2D fixed altitude).

Everything is derived from a few inputs so the scale stays self-consistent:
  bomb coverage  : 250 m2  (Spider-i UAV's H300: 200-300 m2 per mission; one bomb = the full 100 kg load)
  drop footprint : 3x3 cells (equivalent to one drone move of 3 cells, so drops tile the map)
  drone speed    : up to 20 m/s (physical); 3 cells per env step - 15.81 m per env step - 0.79 seconds
  grid           : 240x160 (framework default)

Battery is calibrated from the H300's published 45 min UNLOADED endurance (no power formula).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

FT_PER_M = 3.28084

@dataclass
class DroneConfig:
    # --- scale (derived) ---
    bomb_area_m2: float = 250.0
    footprint_cells: int = 3           # drop = (footprint_cells x footprint_cells) block
    cells_per_step: int = 3            # framework HELICOPTER_SPEED
    speed_mps: float = 20.0
    grid_w: int = 240
    grid_h: int = 160
    max_steps: int = 2000

    # --- H300 airframe / payload ---
    empty_kg: float = 107.8
    payload_kg: float = 100.0
    payload_drops: int = 1

    # --- battery ---
    endurance_empty_min: float = 45.0
    derate: float = 0.8
    battery_scale: float = 1.0

    # --- thermal IR sensor (square, north-up; drone state is 2D) ---
    # Half-side r in cells: the view is (2r+1)x(2r+1) centered on the drone. r=6 -> 13x13 (~69 m at 5.27 m/cell).
    sensor_radius: int = 6

    # --- mission ---
    # Drone deployed near the fire: base sits base_offset_cells west of FireCastRL's map-center spark
    # (25 cells ~ 130 m, outside the sensor view, so the drone must still detect, approach and aim).
    base_offset_cells: int = 25
    base_coord: Optional[tuple] = None

    # --- fire clock / burn ---
    cell_burn_min: float = 3.0
    max_burn_time_min: float = 500.0

    def __post_init__(self):
        if self.base_coord is None:
            spark_x = self.grid_w // 2 - 1  # FireCastRL lights the cell at the map center: (119, 79) on 240x160
            self.base_coord = (spark_x - self.base_offset_cells, self.grid_h // 2)

    # ---------- derived scale ----------
    @property
    def cell_m(self) -> float:
        return math.sqrt(self.bomb_area_m2) / self.footprint_cells

    @property
    def cell_ft(self) -> float:
        return self.cell_m * FT_PER_M

    @property
    def step_m(self) -> float:
        return self.cells_per_step * self.cell_m

    @property
    def dt_s(self) -> float:
        return self.step_m / self.speed_mps

    @property
    def minute_per_timestep(self) -> float:
        return self.dt_s / 60.0               

    def footprint(self) -> list[tuple[int, int]]:
        r = self.footprint_cells // 2
        return [(dx, dy) for dy in range(-r, r + 1) for dx in range(-r, r + 1)]

    # ---------- thermal sensor ----------
    def sensor_offsets(self) -> list[tuple[int, int]]:
        """(dx, dy) offsets of the cells inside the square sensor footprint (axis-aligned, north-up)."""
        r = self.sensor_radius
        return [(dx, dy) for dy in range(-r, r + 1) for dx in range(-r, r + 1)]

    # ---------- battery ----------
    def mass_kg(self, loaded: bool) -> float:
        return self.empty_kg + (self.payload_kg if loaded else 0.0)

    def endurance_min(self, loaded: bool) -> float:
        """Power assumed linear in mass (replace with measured loaded data when found)."""
        return self.endurance_empty_min * self.empty_kg / self.mass_kg(loaded)

    def _frac_per_s(self, loaded: bool) -> float:
        return self.battery_scale / (self.endurance_min(loaded) * 60.0 * self.derate)

    def move_cost(self, loaded: bool) -> float:
        return self._frac_per_s(loaded) * self.dt_s

    # ---------- plumbing into the framework ----------
    def apply_scale(self, config) -> None:
        """Mutate the framework's config module BEFORE constructing the env.

        Values are set directly (not via ceil(model/cell)) to avoid float round-off
        turning 160 rows into 161.
        """
        config.gridWidth = self.grid_w
        config.gridHeight = self.grid_h
        config.cellSize = self.cell_ft
        config.modelWidth = self.grid_w * self.cell_ft
        config.modelHeight = self.grid_h * self.cell_ft
        config.heightmapMaxElevation = 10000.0 * config.modelWidth / 120000.0
        config.minCellBurnTime = self.cell_burn_min
        config.helitackDropRadius = self.cell_ft            
        config.HELICOPTER_SPEED = self.cells_per_step
        config.MAX_TIMESTEPS = self.max_steps
        config.maxTimeStep = self.minute_per_timestep

    def wrapper_kwargs(self, spread_cap: float = 1e3) -> dict:
        return dict(
            ignition_cap=self.minute_per_timestep * self.max_steps,
            spread_cap=spread_cap,                            
            burn_time_cap=self.max_burn_time_min,                              
            elevation_cap=10000.0 * (self.grid_w * self.cell_ft) / 120000.0,
            helitack_cap=float(self.payload_drops),
        )

    def summary(self) -> str:
        return "\n".join([
            f"Cell: {self.cell_ft:.2f} ft = {self.cell_m:.2f} m ({self.cell_m**2:.1f} m2)",
            f"Grid: {self.grid_w}x{self.grid_h} = {self.grid_w*self.cell_ft:.0f} x {self.grid_h*self.cell_ft:.0f} ft ({self.grid_w*self.cell_m:.0f} x {self.grid_h*self.cell_m:.0f} m)",
            f"drop tile: {self.footprint_cells}x{self.footprint_cells} = {self.footprint_cells**2*self.cell_m**2:.0f} m^2 | one move = {self.cells_per_step} cells = "
            f"{self.step_m:.1f} m ({self.step_m*FT_PER_M:.1f} ft)",
            f"step = {self.dt_s:.3f} s = {self.minute_per_timestep:.5f} min",
            f"episode: {self.max_steps} steps = {self.max_steps*self.minute_per_timestep:.1f} min",
            f"weight: loaded {self.mass_kg(True):.1f} kg, empty {self.mass_kg(False):.1f}",
            f"endurance: loaded {self.endurance_min(True):.1f} min, empty {self.endurance_min(False):.1f} min",
            f"battery cost: loaded move {self.move_cost(True):.5f}, empty move {self.move_cost(False):.5f}",
            f"loaded moves to empty {1/self.move_cost(True):.0f}",
            f"thermal sensor: {2*self.sensor_radius+1}x{2*self.sensor_radius+1} cells = "
            f"{(2*self.sensor_radius+1)*self.cell_m:.0f} m square ({len(self.sensor_offsets())} cells in view)",
        ])


if __name__ == "__main__":
    print(DroneConfig().summary())
