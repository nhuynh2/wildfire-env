# FireCastRL – Drone Wildfire Environment

A custom Gymnasium environment, `firecastrl/Drone-env0`, built on top of [FireCastRL](https://github.com/aisystems-lab/wildfire-env). The firefighting helicopter is replaced by a drone (Spider-i H300 class) with one retardant bomb, a limited battery, and a base for refill and recharge.

The code is in the [`wildfire-env/`](wildfire-env/) folder, on the `custom-env` branch of [nhuynh2/wildfire-env](https://github.com/nhuynh2/wildfire-env/tree/custom-env). The original framework documentation is in [wildfire-env/README.md](wildfire-env/README.md).

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
```

Build the 3D viewer. Skip this if you won't use `render_mode="3d"` and haven't changed anything in `web/`.

```bash
cd web
npm install
npm run build
cd ..
```

## Run the tests

Run all commands from the `wildfire-env/` folder.

**Drone environment: random actions, recorded to MP4**

```bash
uv run python scripts/random_drone_video.py --episodes 2
```

This saves `videos/random_drone_ep0.mp4`, `videos/random_drone_ep1.mp4`, and so on, and prints each episode's metrics: cells quenched, drops, wasted drops, refills, cells burnt, whether the drone crashed, and steps. Options: `--seed`, `--fps`, `--out`.

**Optional: Test the framework's original helicopter environment with random actions**

```bash
uv run python scripts/random_agent_human.py   # Pygame window
uv run python scripts/random_agent_3d.py      # 3D viewer in the browser
```

## Custom files

| File                                    | Purpose                                                                                                                                     |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `firecastrl_env/envs/drone_env.py`    | `DroneWildfireEnv`: payload, 3×3 drop, battery, base refill, termination, episode metrics                                                |
| `firecastrl_env/envs/drone_config.py` | `DroneConfig`: map scale and battery costs from H300 specs. Run `uv run python -m firecastrl_env.envs.drone_config` to print the values |
| `firecastrl_env/envs/drone_reward.py` | `RewardCoeffs` (reward weights) and the `drone_reward()` function                                                                       |
| `scripts/random_drone_video.py`       | Random-action test that records videos                                                                                                      |
