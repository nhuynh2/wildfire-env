import argparse
import pathlib
import sys

import gymnasium as gym
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import firecastrl_env

ENV_ID = "firecastrl/Drone-env0"
MODEL_PATH = REPO_ROOT / "models" / "drone_ppo_detect"
from typing import Any, cast
import imageio.v2 as imageio

VIDEO_DIR = REPO_ROOT / "videos"
TB_DIR = REPO_ROOT / "tb_logs"
LOG_DIR = REPO_ROOT / "logs" / "detect"


def make_env(randomize_spawn: bool = True, render_mode: str | None = None):
    return gym.make(
        ENV_ID,
        stage="detect",
        max_steps=150,
        randomize_spawn=randomize_spawn,
        render_mode=render_mode,
    )


def evaluate(
    model: MaskablePPO,
    n_episodes: int = 20,
    randomize_spawn: bool = True,
    record_episodes: int = 3,
    fps: int = 20,
) -> None:
    env = make_env(randomize_spawn=randomize_spawn, render_mode="rgb_array")
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    tag = "random" if randomize_spawn else "base"
    detected, steps, rewards = [], [], []

    for ep in range(n_episodes):
        record = ep < record_episodes
        writer = (
            cast(Any, imageio.get_writer(VIDEO_DIR / f"detect_{tag}_ep{ep}.mp4", fps=fps, macro_block_size=8))
            if record else None
        )
        obs, _ = env.reset(seed=100 + ep)
        if writer:
            writer.append_data(env.render())

        done = truncated = False
        ep_rew, n_steps, saw_fire = 0.0, 0, False
        while not (done or truncated):
            action, _ = model.predict(obs, action_masks=get_action_masks(env), deterministic=True)
            obs, rew, done, truncated, info = env.step(int(action))
            if writer:
                writer.append_data(env.render())
            ep_rew += float(rew)
            n_steps += 1
            saw_fire = saw_fire or bool(info.get("fire_in_view", False))

        if writer:
            writer.close()
        detected.append(saw_fire)
        steps.append(n_steps)
        rewards.append(ep_rew)

    env.close()
    print(f"[Detect | {tag}] detection_rate={np.mean(detected):.2f} | "
          f"avg_steps={np.mean(steps):.1f} | avg_reward={np.mean(rewards):.2f}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=80_000)
    p.add_argument("--n-envs", type=int, default=2)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    # MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    check_env(make_env().unwrapped, warn=False)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    env = make_vec_env(
        make_env,
        n_envs=args.n_envs,
        vec_env_cls=SubprocVecEnv,
        seed=args.seed,
        monitor_dir=str(LOG_DIR),
    )
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=dict(normalize_images=False),
        tensorboard_log=str(TB_DIR),
        n_steps=512,
        batch_size=256,
        verbose=1,
        device=args.device,
        seed=args.seed,
    )
    model.learn(total_timesteps=args.timesteps, tb_log_name="drone_detect")

    # loaded = MaskablePPO.load(MODEL_PATH, device=args.device)
    # evaluate(loaded, n_episodes=5, randomize_spawn=False)
    # evaluate(loaded, n_episodes=20, randomize_spawn=True)


if __name__ == "__main__":
    main()