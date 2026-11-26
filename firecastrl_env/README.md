# FireCastRL – Drone Wildfire Environment

A custom Gymnasium environment, `firecastrl/Drone-env0`, built on top of [FireCastRL](https://github.com/aisystems-lab/wildfire-env). The firefighting helicopter is replaced by a drone (Spider-i H300 class) with one retardant bomb, a limited battery, a base for refill and recharge, and a thermal IR camera that only sees fire close to the drone.

The code is in the [`wildfire-env/`](wildfire-env/) folder, on the `custom-env` branch of [nhuynh2/wildfire-env](https://github.com/nhuynh2/wildfire-env/tree/custom-env). The original framework documentation is in [wildfire-env/README.md](README.md).

## Requirements

- Python 3.11.11
- [`uv`](https://docs.astral.sh/uv/)
- Node.js and npm (only needed to build the 3D browser viewer)

## Setup

```bash
git clone -b custom-env https://github.com/nhuynh2/wildfire-env.git
cd wildfire-env
uv venv --python 3.11.11
uv sync
uv pip install imageio imageio-ffmpeg   # needed by the video-recording scripts
```

`uv sync` removes packages that aren't declared in `pyproject.toml`, so rerun the `uv pip install` line after any `uv sync`.

Build the 3D viewer. Skip this if you won't use `render_mode="3d"` and haven't changed anything in `web/`.

```bash
cd web
npm install
npm run build
cd ..
```

## The task

Every episode starts the same way. The fire ignites where FireCastRL ignites it, at the center of the real map (cell 119, 79). The drone starts at its base 25 cells (about 130 m) west of the fire. The drone has to find the fire with its camera, fly over it, and drop its one bomb on a burning cell.

| | |
| --- | --- |
| **Actions** | `Discrete(5)`: 0 = +y, 1 = −y, 2 = −x, 3 = +x (each move is 3 cells), 4 = drop |
| **Observation** | `local`: 4×13×13 thermal view centered on the drone (burning, burnt, nonburnable, off-map). `position`: x, y scaled to 0–1. `battery`: 0–1. `time`: step / 2000 |
| **Sensor** | Radius 6 cells (≈33 m): a 45° thermal camera at 100 m altitude |
| **Drop** | Released only if a burning cell is in the camera view; otherwise the drone just hovers. A release covers 3×3 cells and ends the episode |
| **Reward** | +1 per burning cell put out, +10 for a hit (a release that lands on fire), −5 for a missed release, −10 for a crash, −0.002 per step, and a small penalty below 25% battery |
| **Episode ends** | Bomb released, battery empty away from base (crash), fire out, or 2000 steps |

## Train PPO

Run all commands from the `wildfire-env/` folder.

```bash
uv run python scripts/train_ppo_drone.py
```

This trains PPO from Stable-Baselines3 (`MultiInputPolicy`, default hyperparameters, as in the framework README) on 4 parallel copies of the environment. When training finishes, it:

1. saves the model to `models/ppo_drone.zip`
2. plays the first 3 episodes with the trained agent and saves each one to `videos/ppo_drone_ep0.mp4`, `videos/ppo_drone_ep1.mp4`, and so on
3. prints each episode's result and a summary: success rate, detection rate, steps to detection and to release, wasted-release rate, crash rate, cells burnt, and suppression efficiency

| Option | Default | Meaning |
| --- | --- | --- |
| `--timesteps` | `100000` | Total training steps |
| `--n-envs` | `4` | Parallel environments (only makes training faster) |
| `--device` | `mps` | `mps` (Apple GPU) or `cpu`. The simulator is the bottleneck, so `cpu` is about as fast |
| `--episodes` | `3` | Episodes recorded after training |
| `--fps` | `20` | Video playback speed (environment steps per second) |

Example: a longer run on the CPU that records 5 episodes:

```bash
uv run python scripts/train_ppo_drone.py --timesteps 300000 --device cpu --episodes 5
```

100k steps take roughly 4–5 minutes on an Apple-silicon laptop. Each run overwrites `models/ppo_drone.zip` and the `ppo_drone_ep*.mp4` videos.

## Run the tests

**Drone environment: random actions, recorded to MP4**

```bash
uv run python scripts/random_drone_video.py --episodes 2
```

This saves `videos/random_drone_ep0.mp4`, `videos/random_drone_ep1.mp4`, and so on, and prints each episode's metrics: cells put out, releases, blocked drops, whether the drop hit, whether and when the fire was detected, cells burnt, whether the drone crashed, and steps. Options: `--seed`, `--fps`, `--out`.

**Print the drone's scale, battery, and sensor values**

```bash
uv run python -m firecastrl_env.envs.drone_config
```

**Optional: Test the framework's original helicopter environment with random actions**

```bash
uv run python scripts/random_agent_human.py   # Pygame window
uv run python scripts/random_agent_3d.py      # 3D viewer in the browser
```

## Custom files

| File | Purpose |
| --- | --- |
| `firecastrl_env/envs/drone_env.py` | `DroneWildfireEnv`: thermal-camera observation, drop interlock, payload, 3×3 drop, battery, base refill, termination, episode metrics |
| `firecastrl_env/envs/drone_config.py` | `DroneConfig`: map scale and battery costs from H300 specs, camera altitude and field of view, base position |
| `firecastrl_env/envs/drone_reward.py` | `RewardCoeffs` (reward weights) and the `drone_reward()` function |
| `scripts/train_ppo_drone.py` | Trains PPO, saves the model, records episodes of the trained agent |
| `scripts/random_drone_video.py` | Random-action test that records videos |
