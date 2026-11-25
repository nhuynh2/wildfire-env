from dataclasses import dataclass


@dataclass
class RewardCoeffs:
    """
    Task 1 max single-step reward: 10.0 (detect) + 0.0615 (explore) - 0.05 (step) = +10.01
    Task 2 max single-step reward: 2.0 (hit) + 13.0 (all 13 cells quenched) - 0.05 (step) = +14.95
    Min single-step reward: -2.0 (wasted) - 0.05 (step) = -2.05 
    """

    # --- Base eval fallback & clipping ---
    step: float = 0.01
    clip: float = 20.0

    # --- Task 1 Search & Detect Rewards ---
    detect_bonus: float = 10.0     # terminal reward when any burning cell enters 41x41 view
    explore_cell: float = 0.0005   # per newly uncovered cell inside 41x41 view

    # --- Task 2 / Base Drop Rewards ---
    quench: float = 1.0            # per burning cell extinguished (max 13 -> +13.0)
    hit: float = 2.0               # bonus when release lands on >= 1 burning cell
    wasted: float = 2.0            # penalty when release hits 0 burning cells
    approach: float = 0.25         # potential-based shaping per cell closer to nearest burning cell


def drone_reward(c: RewardCoeffs, *, quenched: int, released: bool) -> tuple[float, dict[str, float]]:
    """Base reward used in stage='eval'."""
    hit = released and quenched > 0
    wasted = released and quenched == 0
    terms = {
        "quench": c.quench * quenched,
        "hit": c.hit * float(hit),
        "wasted": -c.wasted * float(wasted),
        "step": -c.step,
    }
    total = sum(terms.values())
    return float(max(-c.clip, min(c.clip, total))), terms


def detect_reward(c: RewardCoeffs, *, detected: bool, newly_seen: int) -> tuple[float, dict[str, float]]:
    """Task 1 (Search & Detect) reward."""
    terms = {
        "detect": c.detect_bonus * float(detected),
        "explore": c.explore_cell * float(newly_seen),
        "step": -c.step,
    }
    total = sum(terms.values())
    return float(max(-c.clip, min(c.clip, total))), terms


def drop_reward(
    c: RewardCoeffs,
    *,
    quenched: int,
    released: bool,
    dist_delta: float,
) -> tuple[float, dict[str, float]]:
    """Task 2 (Target & Drop) reward."""
    hit = released and quenched > 0
    wasted = released and quenched == 0
    terms = {
        "quench": c.quench * quenched,
        "hit": c.hit * float(hit),
        "wasted": -c.wasted * float(wasted),
        "approach": c.approach * float(dist_delta) if not released else 0.0,
        "step": -c.step,
    }
    total = sum(terms.values())
    return float(max(-c.clip, min(c.clip, total))), terms