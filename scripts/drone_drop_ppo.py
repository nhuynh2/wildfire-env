"""Task 2 (Target & Drop): train and evaluate MaskablePPO.

Each episode pre-ticks the fire 10-50 times, then spawns the drone with fire in view
(within +/-18 cells of a random burning cell). The episode ends on the single release or after 50 steps.

    python scripts/drone_drop_ppo.py                 # train, save, then evaluate the final model
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
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

import firecastrl_env  # noqa: F401  (registers the env ids)

ENV_ID = "firecastrl/Drone-env0"
MODEL_PATH = REPO_ROOT / "models" / "drone_ppo_drop"
DETECT_BEST = REPO_ROOT / "models" / "detect_best" / "best_model.zip"  # best detect eval: step 97,280
VIDEO_DIR = REPO_ROOT / "videos"
TB_DIR = REPO_ROOT / "tb_logs"
LOG_DIR = REPO_ROOT / "logs" / "drop"
EVAL_SEED = 1000  # episodes used by the eval callback during training
TEST_SEED = 2000  # held-out episodes for evaluate() after training


def make_env(render_mode: str | None = None):
    return gym.make(ENV_ID, stage="drop", max_steps=50, render_mode=render_mode)


class PerEpisodeSeed(gym.Wrapper):
    """After reset(seed=s), later unseeded resets use s+1, s+2, ...

    Drop episodes end at the release, so their length depends on the policy. Without this, one seed per
    eval round would leave episodes 2..N depending on how many random draws earlier episodes used.
    """

    def __init__(self, env):
        super().__init__(env)
        self._base: int | None = None
        self._i = 0

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._base, self._i = seed, 0
        s = None if self._base is None else self._base + self._i
        self._i += 1
        return self.env.reset(seed=s, options=options)


class DropEvalCallback(MaskableEvalCallback):
    """MaskableEvalCallback on the same episodes every round, plus drop-task metrics.

    Logs eval/mean_reward, eval/mean_ep_length (from the parent), and:
      eval/success_rate               release hit >= 1 burning cell
      eval/mean_steps_to_release      released episodes only
      eval/mean_suppression_efficiency  quenched cells / drop capacity (13), all episodes
    """

    def __init__(self, *args, eval_seed: int = EVAL_SEED, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_seed = eval_seed
        self._release_buf: list[int] = []
        self._eff_buf: list[float] = []

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self.eval_env.seed(self.eval_seed)
            self._release_buf, self._eff_buf = [], []
        return super()._on_step()

    def _log_success_callback(self, locals_, globals_) -> None:
        if not locals_["done"]:
            return
        m = locals_["info"].get("episode_metrics")
        if m is None:
            return
        self._is_success_buffer.append(bool(m["is_success"]))
        self._eff_buf.append(float(m["suppression_efficiency"]))
        if m["steps_to_release"] is not None:
            self._release_buf.append(int(m["steps_to_release"]))
        self.logger.record("eval/mean_suppression_efficiency", float(np.mean(self._eff_buf)))
        if self._release_buf:
            self.logger.record("eval/mean_steps_to_release", float(np.mean(self._release_buf)))


def evaluate(
    model: MaskablePPO,
    n_episodes: int = 50,
    record_episodes: int = 3,
    fps: int = 10,
    seed: int = TEST_SEED,
) -> None:
    env = make_env(render_mode="rgb_array")
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    success, release_steps, eff, rewards = [], [], [], []

    for ep in range(n_episodes):
        writer = (
            cast(Any, imageio.get_writer(VIDEO_DIR / f"drop_ep{ep}.mp4", fps=fps, macro_block_size=8))
            if ep < record_episodes else None
        )
        obs, _ = env.reset(seed=seed + ep)
        if writer:
            writer.append_data(env.render())

        done = truncated = False
        ep_rew, info = 0.0, {}
        while not (done or truncated):
            action, _ = model.predict(obs, action_masks=get_action_masks(env), deterministic=True)
            obs, rew, done, truncated, info = env.step(int(action))
            if writer:
                writer.append_data(env.render())
            ep_rew += float(rew)
        if writer:
            writer.close()

        m = info["episode_metrics"]
        success.append(bool(m["is_success"]))
        eff.append(float(m["suppression_efficiency"]))
        if m["steps_to_release"] is not None:
            release_steps.append(int(m["steps_to_release"]))
        rewards.append(ep_rew)

    env.close()
    mean_release = f"{np.mean(release_steps):.1f}" if release_steps else "n/a"
    print(f"[Drop] success_rate={np.mean(success):.2f} | steps_to_release={mean_release} | "
          f"suppression_efficiency={np.mean(eff):.2f} | avg_reward={np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=81_920)
    p.add_argument("--n-envs", type=int, default=10)
    p.add_argument("--n-steps", type=int, default=128)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-freq", type=int, default=10_240, help="timesteps between evaluations")
    p.add_argument("--eval-episodes", type=int, default=30)
    p.add_argument(
        "--init-from",
        default=str(DETECT_BEST),
        help="policy weights to start drop training from (default: best detect model); 'none' = from scratch",
    )
    args = p.parse_args()

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
        n_steps=args.n_steps,
        batch_size=256,
        verbose=1,
        device=args.device,
        seed=args.seed,
    )
    if args.init_from.lower() != "none":
        # Same observation/action spaces and network as detect, so the weights load as-is (policy + value
        # heads + optimizer state). Hyperparameters above (n_steps, n_envs, ...) are this script's, not detect's.
        model.set_parameters(args.init_from, exact_match=True, device=args.device)
        print(f"Initialized drop policy from {args.init_from}")
    eval_env = make_vec_env(lambda: PerEpisodeSeed(make_env()), n_envs=1)
    eval_cb = DropEvalCallback(
        eval_env,
        n_eval_episodes=args.eval_episodes,
        eval_freq=max(1, args.eval_freq // args.n_envs),
        deterministic=True,
        log_path=str(LOG_DIR / "eval"),
        best_model_save_path=str(MODEL_PATH.parent / "drop_best"),
        verbose=1,
    )
    try:
        model.learn(total_timesteps=args.timesteps, tb_log_name="drone_drop", callback=eval_cb)
    except KeyboardInterrupt:
        print(f"Interrupted at {model.num_timesteps} timesteps; saving the model trained so far.")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Saved final model to {MODEL_PATH}.zip")
    for e in (env, eval_env):
        try:
            e.close()
        except (EOFError, BrokenPipeError, ConnectionResetError):
            pass

    loaded = MaskablePPO.load(MODEL_PATH, device=args.device)
    evaluate(loaded, n_episodes=args.eval_episodes, record_episodes=3)


if __name__ == "__main__":
    main()
