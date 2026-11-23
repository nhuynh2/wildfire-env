from dataclasses import dataclass


@dataclass
class RewardCoeffs:
    quench: float = 0.0                 # per burning cell extinguished by the release
    hit: float = 0.0                   # main goal: the release lands on >= 1 burning cell
    wasted: float = 0.0                 # release (fire in view) that misses every burning cell
    step: float = 0.01                  # per-step time cost
    clip: float = 0.0 * quench + 1                  # must exceed hit + max quench (10 + 9)

def drone_reward(c: RewardCoeffs, *, quenched, released):
    total = 0.0
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
