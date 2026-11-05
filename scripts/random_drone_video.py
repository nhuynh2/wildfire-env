import argparse
import pathlib
import sys

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import firecastrl_env  # noqa: F401


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run random drone episodes and record them (top-down rgb_array render) to an MP4."
    )
    parser.add_argument("--env-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=3, help="Each episode is recorded to its own video.")
    parser.add_argument("--fps", type=int, default=20, help="Playback speed of the video (env steps per second).")
    parser.add_argument("--out", type=pathlib.Path, default=REPO_ROOT / "videos", help="Directory for the videos.")
    args = parser.parse_args()

    env = gym.make("firecastrl/Drone-env0", env_id=args.env_id, render_mode="rgb_array")
    rng = np.random.default_rng(args.seed)
    drop_prob = 0.01
    probs = [(1 - drop_prob) / 4] * 4 + [drop_prob]   # [0.2475, 0.2475, 0.2475, 0.2475, 0.01]

    args.out.mkdir(parents=True, exist_ok=True)
    try:
        for episode in range(args.episodes):
            path = args.out / f"random_drone_ep{episode}.mp4"
            with imageio.get_writer(path, fps=args.fps, macro_block_size=8) as video:
                env.reset(seed=args.seed + episode)
                video.append_data(env.render())

                terminated = truncated = False
                info = {}
                while not (terminated or truncated):
                    action = int(rng.choice(5, p=probs))
                    _, _, terminated, truncated, info = env.step(action)
                    video.append_data(env.render())

            print(f"episode {episode}: {info.get('episode_metrics')}")
            print(f"saved {path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
