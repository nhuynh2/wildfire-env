import argparse
import pathlib
import sys

import gymnasium as gym
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import firecastrl_env
from firecastrl_env.envs.drone_config_2 import DroneConfig2


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ENV_ID = "firecastrl/Drone-env0"
MODEL_PATH = REPO_ROOT / "models" / "drone_ppo_detect"
from typing import Any, cast
import imageio.v2 as imageio

VIDEO_DIR = REPO_ROOT / "videos"
TB_DIR = REPO_ROOT / "tb_logs"
LOG_DIR = REPO_ROOT / "logs" / "detect"
EVAL_SEED = 1000


def make_env(randomize_spawn: bool = True, render_mode: str | None = None, base_coord=None):
    return gym.make(
        ENV_ID,
        stage="detect",
        max_steps=150,
        randomize_spawn=randomize_spawn,
        render_mode=render_mode,
        drone_cfg=DroneConfig2(base_coord=base_coord)
    )


class DetectEvalCallback(MaskableEvalCallback):
    """MaskableEvalCallback on the same start positions every time, plus detect-task metrics.

    Logs eval/mean_reward, eval/mean_ep_length, eval/success_rate (= detection rate),
    eval/mean_steps_to_detect (detected episodes only).
    """

    def __init__(self, *args, eval_seed: int = EVAL_SEED, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_seed = eval_seed
        self._steps_buf: list[int] = []

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self.eval_env.seed(self.eval_seed)
            self._steps_buf= []
        return super()._on_step()

    def _log_success_callback(self, locals_, globals_) -> None:
        if not locals_["done"]:
            return
        m = locals_["info"].get("episode_metrics")
        if m is None:
            return
        self._is_success_buffer.append(bool(m["detected"]))
        if m["detected"]:
            self._steps_buf.append(m["steps_to_detection"])
        if self._steps_buf:
            self.logger.record("eval/mean_steps_to_detect", float(np.mean(self._steps_buf)))


def evaluate(
    model: MaskablePPO,
    n_episodes: int = 50,
    randomize_spawn: bool = True,
    record_episodes: int = 3,
    fps: int = 10,
    base_coord: tuple[int, int] | None = None,
    seed: int = EVAL_SEED
) -> None:
    env = make_env(randomize_spawn=randomize_spawn, render_mode="rgb_array", base_coord=base_coord)
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    tag = "random" if randomize_spawn else "base"
    detected, steps, rewards, detect_steps = [], [], [], []

    for ep in range(n_episodes):
        record = ep < record_episodes
        writer = (
            cast(Any, imageio.get_writer(VIDEO_DIR / f"detect_{tag}_ep{ep}.mp4", fps=fps, macro_block_size=8))
            if record else None
        )
        obs, _ = env.reset(seed=seed + ep)
        if writer:
            writer.append_data(env.render())

        done = truncated = False
        ep_rew, n_steps, saw_fire = 0.0, 0, False
        info = {}
        while not (done or truncated):
            action, _ = model.predict(obs, action_masks=get_action_masks(env), deterministic=True)
            obs, rew, done, truncated, info = env.step(int(action))
            if writer:
                writer.append_data(env.render())
            ep_rew += float(rew)
            n_steps += 1
            saw_fire = saw_fire or bool(info.get("fire_in_view", False))
        m = info["episode_metrics"]
        if m["detected"]:
            detect_steps.append(m["steps_to_detection"])

        if writer:
            writer.close()
        detected.append(saw_fire)
        steps.append(n_steps)
        rewards.append(ep_rew)

    env.close()
    mean_detect = f"{np.mean(detect_steps):.1f}" if detect_steps else "n/a"
    print(f"[Detect | {tag}] detection_rate={np.mean(detected):.2f} | "
          f"steps_to_detect={mean_detect} | "
          f"avg_steps={np.mean(steps):.1f} | avg_reward={np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=102400)
    p.add_argument("--n-envs", type=int, default=2)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-freq", type=int, default=5120, help="timesteps between evaluations")
    p.add_argument("--eval-episodes", type=int, default=30)
    args = p.parse_args()

    # check_env(make_env().unwrapped, warn=False)

    # LOG_DIR.mkdir(parents=True, exist_ok=True)
    # env = make_vec_env(
    #     make_env,
    #     n_envs=args.n_envs,
    #     vec_env_cls=SubprocVecEnv,
    #     seed=args.seed,
    #     monitor_dir=str(LOG_DIR),
    # )
    # model = MaskablePPO(
    #     "MultiInputPolicy",
    #     env,
    #     policy_kwargs=dict(normalize_images=False),
    #     tensorboard_log=str(TB_DIR),
    #     n_steps=512,
    #     batch_size=256,
    #     verbose=1,
    #     device=args.device,
    #     seed=args.seed,
    # )
    # eval_env = make_vec_env(make_env, n_envs=1)
    # eval_cb = DetectEvalCallback(
    #     eval_env,
    #     n_eval_episodes=args.eval_episodes,
    #     eval_freq=max(1, args.eval_freq // args.n_envs),
    #     deterministic=True,
    #     log_path=str(LOG_DIR / "eval"),
    #     best_model_save_path=str(MODEL_PATH.parent / "detect_best"),
    #     verbose=1,
    # )
    # model.learn(total_timesteps=args.timesteps, tb_log_name="drone_detect", callback=eval_cb)

    # MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    # model.save(MODEL_PATH)
    loaded = MaskablePPO.load(MODEL_PATH, device=args.device)
    # evaluate(loaded, n_episodes=5, randomize_spawn=False)
    evaluate(loaded, n_episodes=1, randomize_spawn=False, base_coord=(15, 25))
    # evaluate(loaded, n_episodes=1, randomize_spawn=False, base_coord=(239, 159))
    # evaluate(loaded, n_episodes=1, randomize_spawn=False, base_coord=(0, 159))
    # evaluate(loaded, n_episodes=1, randomize_spawn=False, base_coord=(239, 0))

if __name__ == "__main__":
    main()