# FireCastRL – Drone Wildfire Environment

A drone firefighting environment, `firecastrl/Drone-env0`, built on top of [FireCastRL](https://github.com/aisystems-lab/wildfire-env). A single MaskablePPO policy is trained in two phases: Task 1 (detect the fire), then Task 2 (drop retardant on it), and is evaluated on the full mission.

For installation, the fire simulator, and the original helicopter environment, see the framework's [README](../README.md).

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
