"""Full-mission evaluation (stage="eval") of the single warm-started policy: start at base, find the fire, drop once.

    python scripts/drone_mission_eval.py
    python scripts/drone_mission_eval.py --drone-coord 239 0 --record 1
"""
import argparse
import pathlib
import sys
from typing import Any, cast

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
from sb3_contrib import MaskablePPO

import firecastrl_env  # noqa: F401  (registers the env ids)

ENV_ID = "firecastrl/Drone-env0"
MODEL = REPO_ROOT / "models" / "drone_ppo_drop.zip"
VIDEO_DIR = REPO_ROOT / "videos"
TEST_SEED = 3000
FULL_MASK = np.ones(5, dtype=bool)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(MODEL))
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--seed", type=int, default=TEST_SEED)
    p.add_argument("--max-steps", type=int, default=300, help="give up after this many steps (env limit is 2000)")
    p.add_argument("--drone-coord", type=int, nargs=2, default=None, help="start here instead of the base")
    p.add_argument("--record", type=int, default=3, help="number of episodes to save as video")
    p.add_argument("--fps", type=int, default=10)
    args = p.parse_args()

    model = MaskablePPO.load(args.model, device="cpu")
    env = gym.make(ENV_ID, stage="eval", render_mode="rgb_array" if args.record else None)
    options = {"drone_coord": tuple(args.drone_coord)} if args.drone_coord else None
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for ep in range(args.episodes):
        writer = (
            cast(Any, imageio.get_writer(VIDEO_DIR / f"mission_ep{ep}.mp4", fps=args.fps, macro_block_size=8))
            if ep < args.record else None
        )
        obs, _ = env.reset(seed=args.seed + ep, options=options)
        if writer:
            writer.append_data(env.render())

        done = truncated = False
        info: dict = {}
        steps = 0
        while not (done or truncated) and steps < args.max_steps:
            a, _ = model.predict(obs, action_masks=FULL_MASK, deterministic=True)
            obs, _, done, truncated, info = env.step(int(a))
            steps += 1
            if writer:
                writer.append_data(env.render())
        if writer:
            writer.close()

        m = info.get("episode_metrics")
        if m is None:  # hit --max-steps before the env ended the episode
            first_detect = getattr(env.unwrapped, "_first_detect_step", None)
            m = {"is_success": False, "detected": first_detect is not None, "steps_to_detection": first_detect,
                 "steps_to_release": None, "suppression_efficiency": 0.0, "released": 0}
        rows.append({**m, "timed_out": not (done or truncated)})

    env.close()

    def mean(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return f"{np.mean(vals):.1f}" if vals else "n/a"

    print(f"[Mission] episodes={len(rows)} start={args.drone_coord or 'base'}")
    print(f"  success_rate           {np.mean([r['is_success'] for r in rows]):.2f}  (drop hit >= 1 burning cell)")
    print(f"  detection_rate         {np.mean([r['detected'] for r in rows]):.2f}")
    print(f"  release_rate           {np.mean([r['released'] > 0 for r in rows]):.2f}")
    print(f"  timeout_rate           {np.mean([r['timed_out'] for r in rows]):.2f}  (no release within {args.max_steps} steps)")
    print(f"  steps_to_detection     {mean('steps_to_detection')}")
    print(f"  steps_to_release       {mean('steps_to_release')}")
    print(f"  suppression_efficiency {np.mean([r['suppression_efficiency'] for r in rows]):.2f}  (quenched / 13)")


if __name__ == "__main__":
    main()
