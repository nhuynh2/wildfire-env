from dataclasses import dataclass


@dataclass
class RewardCoeffs:
    quench: float = 1.0                 # per burning cell extinguished by the release
    hit: float = 10.0                   # main goal: the release lands on >= 1 burning cell
    wasted: float = 5.0                 # release (fire in view) that misses every burning cell
    crash: float = 10.0                 # battery depleted away from base
    step: float = 0.002                 # per-step time cost
    low_battery: float = 0.05           # max per-step penalty, ramps up below the threshold (battery mode)
    low_battery_threshold: float = 0.25
    clip: float = 20.0                  # must exceed hit + max quench (10 + 9)


def drone_reward(c: RewardCoeffs, *, quenched, released, crashed, battery=1.0):
    """Interlocked drops (no fire in view) are hovers: their cost is battery + step, no extra term."""
    hit = released and quenched > 0
    wasted = released and quenched == 0
    low = max(0.0, (c.low_battery_threshold - battery) / c.low_battery_threshold)
    terms = {
        "quench": c.quench * quenched,
        "hit": c.hit * float(hit),
        "wasted": -c.wasted * float(wasted),
        "crash": -c.crash * float(crashed),
        "step": -c.step,
        "low_battery": -c.low_battery * low,
    }
    total = sum(terms.values())
    return float(max(-c.clip, min(c.clip, total))), terms
