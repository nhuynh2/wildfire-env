"""Train PPO on the drone env, same recipe as the README's Stable-Baselines3 example.

    python scripts/train_ppo_drone.py                      # train, save, record 3 episodes
    python scripts/train_ppo_drone.py --timesteps 300000   # longer run

The fire starts where FireCastRL starts it (the center of the real map); the drone starts at base, 25 cells
west of it. Both are the same every episode. After training, the first --episodes episodes of the trained
agent are recorded (top-down rgb_array render) to videos/ppo_drone_ep{N}.mp4, as in random_drone_video.py.
"""
import argparse
import pathlib
import sys
from typing import Any, cast

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import firecastrl_env  # noqa: F401
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

ENV_ID = "firecastrl/Drone-env0"
MODEL_PATH = REPO_ROOT / "models" / "ppo_drone"
VIDEO_DIR = REPO_ROOT / "videos"


def evaluate(model, n_episodes: int, fps: int) -> None:
    env = gym.make(ENV_ID, render_mode="rgb_array")
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for ep in range(n_episodes):
        path = VIDEO_DIR / f"ppo_drone_ep{ep}.mp4"
        with cast(Any, imageio.get_writer(path, fps=fps, macro_block_size=8)) as video:
            obs, info = env.reset(seed=ep)
            video.append_data(env.render())
            done = truncated = False
            while not (done or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(int(action))
                video.append_data(env.render())
        m = info["episode_metrics"]
        results.append(m)
        print(f"episode {ep}: success={m['success']} detected_at={m['steps_to_detection']} "
              f"released_at={m['steps_to_release']} crashed={m['crashed']} cells_burnt={m['cells_burnt']}")
        print(f"saved {path}")
    env.close()

    def rate(key):
        return np.mean([bool(m[key]) for m in results])

    def avg(key):
        vals = [m[key] for m in results if m[key] is not None]
        return np.mean(vals) if vals else float("nan")

    print(f"\nsuccess rate {rate('success'):.2f} | detection rate {rate('detected'):.2f} | "
          f"steps to detection {avg('steps_to_detection'):.0f} | steps to release {avg('steps_to_release'):.0f}")
    print(f"wasted-release rate {rate('wasted_release'):.2f} | crash rate {rate('crashed'):.2f} | "
          f"cells burnt {avg('cells_burnt'):.1f} | suppression efficiency {avg('suppression_efficiency'):.2f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=300000)
    p.add_argument("--n-envs", type=int, default=2, help="parallel envs (only speeds up training)")
    p.add_argument("--device", default="mps", help="mps or cpu")
    p.add_argument("--episodes", type=int, default=3, help="episodes recorded after training (one video each)")
    p.add_argument("--fps", type=int, default=20, help="playback speed of the videos (env steps per second)")
    args = p.parse_args()

    check_env(gym.make(ENV_ID).unwrapped, warn=False)
    env = make_vec_env(ENV_ID, n_envs=args.n_envs, vec_env_cls=SubprocVecEnv)
    model = PPO("MultiInputPolicy", env, verbose=1, device=args.device)
    model.learn(total_timesteps=args.timesteps)
    model.save(MODEL_PATH)
    env.close()
    print(f"saved {MODEL_PATH}.zip")

    model = PPO.load(MODEL_PATH, device=args.device)
    evaluate(model, args.episodes, args.fps)


if __name__ == "__main__":
    main()
