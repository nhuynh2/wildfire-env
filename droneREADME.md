# FireCastRL – Drone Wildfire Environment

A drone firefighting environment, `firecastrl/Drone-env0`, built on top of [FireCastRL](https://github.com/aisystems-lab/wildfire-env). A single MaskablePPO policy is trained in two phases: Task 1 (detect the fire), then Task 2 (drop retardant on it), and is evaluated on the full mission.

For installation, the fire simulator, and the original helicopter environment, see the framework's [README](https://github.com/nhuynh2/wildfire-env/blob/1fa1cc84da2de517e6c406f580422b8abb75787b/README.md).

## The task

| | |
| --- | --- |
| **Actions** | `Discrete(5)`: 0 = +y, 1 = −y, 2 = −x, 3 = +x (each move is 3 cells), 4 = drop. Drop is masked in Task 1 |
| **Observation** | `local`: 4×41×41 thermal view centered on the drone (burning, burnt, nonburnable, off-map); Task 1 sees the burning channel only. `position`: x, y scaled to 0–1 |
| **Sensor** | Square view, 20 cells in each direction from the drone (41×41 cells) |
| **Drop** | One payload. Released only if a burning cell is in view; otherwise nothing is released. Covers a 13-cell disk (radius 2 cells) and ends the episode |
| **Reward: Task 1** | +10 on detection, +0.0005 per newly scanned cell, −0.01 per step |
| **Reward: Task 2** | +1 per burning cell put out, +2 for a hit, −2 for a drop that hits no fire, +0.25 per cell moved closer to the nearest fire, −0.01 per step |
| **Reward: full mission** | +1 per burning cell put out, +2 for a hit, −2 for a drop that hits no fire, −0.01 per step |
| **Episode ends** | Task 1: fire enters view, or 150 steps. Task 2: drop released, or 50 steps. Full mission: drop released, fire out, or 2,000 steps |

## My files

| File                                      | Purpose                                                                                                                |
| ----------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `firecastrl_env/envs/drone_env.py`      | `DroneWildfireEnv`: detect / drop / eval stages, 41×41 sensor view, action masking, drop interlock, episode metrics |
| `firecastrl_env/envs/drone_config_2.py` | `DroneConfig2`: sensor radius, drop footprint, payload, base position                                                |
| `firecastrl_env/envs/drone_reward.py`   | Reward weights and reward functions for each stage                                                                     |
| `scripts/drone_detect_ppo.py`           | Task 1: train and evaluate the detect policy                                                                           |
| `scripts/drone_drop_ppo.py`             | Task 2: train the drop policy starting from the best detect weights, then evaluate it                                  |
| `scripts/drone_mission_eval.py`         | Full-mission evaluation of the final policy                                                                            |

The environment is registered as `firecastrl/Drone-env0` in `firecastrl_env/__init__.py`.

## Run the scripts

Run all commands from the `wildfire-env/` folder. Videos are saved to `videos/`, models to `models/`.

### Task 1: detect

```bash
uv run python scripts/drone_detect_ppo.py
```

The training block in `main()` is currently commented out, so this loads `models/drone_ppo_detect.zip` and records one evaluation episode. Uncomment the block to retrain (102,400 timesteps, about 27 min). The best checkpoint is saved to `models/detect_best/`.

### Task 2: drop

```bash
uv run python scripts/drone_drop_ppo.py
```

This trains from `models/detect_best/best_model.zip` (81,920 timesteps, 10 envs, about 33 min), saves `models/drone_ppo_drop.zip`, then evaluates it on 30 held-out episodes and records 3 videos. Use `--init-from none` to train from scratch.

### Full mission

```bash
uv run python scripts/drone_mission_eval.py
```

This runs the final policy from the base on 30 episodes, prints success, detection, and suppression metrics, and records 3 videos. Use `--record 1` for one video, or `--drone-coord X Y` to start somewhere other than the base.

## TensorBoard

```bash
tensorboard --logdir tb_logs
```

Open [http://localhost:6006](http://localhost:6006). The runs are `drone_detect_1` and `drone_drop_1`. The main curves are `rollout/ep_rew_mean`, `rollout/ep_len_mean`, and `eval/success_rate`, plus `eval/mean_steps_to_detect` (detect) and `eval/mean_suppression_efficiency` and `eval/mean_steps_to_release` (drop).
