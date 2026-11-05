from dataclasses import dataclass


@dataclass
class RewardCoeffs:
    quench: float = 1.0                 # per burning cell extinguished by the drop (direct result)
    step: float = 0.002                 # per-step time cost
    wasted: float = 0.2                 # last_action == 4 and nothing quenched
    burning: float = 0.01               # per burning cell per step
    low_battery: float = 0.05           # max per-step penalty, ramps up below the threshold (battery mode)
    low_battery_threshold: float = 0.25
    clip: float = 10.0                  # framework convention
 
 
def drone_reward(c: RewardCoeffs, *, quenched, last_action, battery=1.0, burning=0):
    wasted = (last_action == 4) and quenched == 0
    low = max(0.0, (c.low_battery_threshold - battery) / c.low_battery_threshold)
    terms = {
        "quench": c.quench * quenched,
        "step": -c.step,
        "wasted": -c.wasted * float(wasted),
        "burning": -c.burning * burning,
        "low_battery": -c.low_battery * low,
    }
    total = sum(terms.values())
    return float(max(-c.clip, min(c.clip, total))), terms
