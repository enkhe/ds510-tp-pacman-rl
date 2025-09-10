<!-- [![pipeline status](https://gitlab.com/araffin/rl-baselines3-zoo/badges/master/pipeline.svg)](https://gitlab.com/araffin/rl-baselines3-zoo/-/commits/master) -->
![CI](https://github.com/DLR-RM/rl-baselines3-zoo/workflows/CI/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/rl-baselines3-zoo/badge/?version=master)](https://rl-baselines3-zoo.readthedocs.io/en/master/?badge=master)
![coverage report](https://img.shields.io/badge/coverage-68%25-brightgreen.svg?style=flat") [![codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)



# DS510 - Pacman RL Bot player - Customized RL Baselines3 Zoo: A Training Framework for Stable Baselines3 Reinforcement Learning Agents

> Note (DS510 Team Project): This repository is a heavily modified fork of the original RL Baselines3 Zoo, adapted specifically for our DS510 group project. It retains the RL Zoo training core but adds DS510-focused features, Windows-first setup, and Ms. Pac‑Man centric workflows.
>
> Based on: RL Baselines3 Zoo (DLR‑RM) — https://github.com/DLR-RM/rl-baselines3-zoo
>
> Major modifications in this fork:
> - Ms. Pac‑Man GUI training command and Windows PowerShell instructions
> - Enhanced training pipeline (`enhanced_training.py`) with reward shaping, larger CNNs, and callbacks
> - Interactive play and leaderboard (`play_and_rank.py`, `scoreboard.json`)
> - Project-specific docs (`MsPacman_GUI_Training.md`, `MsPacman_RL_Documentation.md`) and plotting utilities
> - Simplified entry points (`train.py`, `enjoy.py`) and course-oriented README

## Open‑source libraries used (acknowledgements)

This project stands on the shoulders of awesome open‑source work. Core dependencies include:

- Stable‑Baselines3 — RL algorithms and policies: https://github.com/DLR-RM/stable-baselines3
- RL Baselines3 Zoo — training framework and tooling: https://github.com/DLR-RM/rl-baselines3-zoo
- Gym / Gymnasium — environment API and Atari envs
  - Default in this fork: Gym 0.26.x (Ms. Pac‑Man NoFrameskip‑v4)
  - Optional: Gymnasium with ALE v5 ids (e.g., `ALE/MsPacman-v5`)
  - Gymnasium extras for Atari: `gymnasium[atari]`, `ale-py`, and `AutoROM` (for ROM installation)
- Environment packages: `box2d-py`, `pybullet_envs_gymnasium`
- Experiment and plotting: `wandb` (optional), `plotly`, `moviepy`, `cloudpickle`, `optunahub`

Links to individual project licenses are available on their respective repositories. See `requirements.txt` for pinned versions used here.

## What’s modified in this DS510 fork (comprehensive)

- Ms. Pac‑Man focus and GUI training
  - Added a documented GUI training path using Gym `render_mode=human` with single-env to avoid multiple windows.
  - Windows-first instructions and quoting guidance for PowerShell.
- Enhanced training pipeline
  - `enhanced_training.py` implements reward shaping via `EnhancedMsPacManWrapper`:
    - survival bonus (+0.01/step), score-delta bonus (`(score_t - score_{t-1})/100`), and life-loss penalty (−10).
  - Larger policy/value MLP heads via `policy_kwargs` (e.g., 512-512-256) on top of SB3 NatureCNN.
  - Adds `EvalCallback`, `CheckpointCallback`, and `StopTrainingOnRewardThreshold`.
  - Supports quick algorithm comparison across PPO/A2C/DQN.
- Play & Rank experience
  - `play_and_rank.py` for human play (arrow keys) or bot playback; persists scores in `scoreboard.json`.
- Plotting & analysis
  - `plot_pacman_loss.py` to export 4K training-loss plots from CSV.
  - Additional plotting helpers under `scripts/` (e.g., TensorBoard-based exporters).
- Convenience & docs
  - Thin entry points: `train.py` and `enjoy.py` wrap RL Zoo.
  - Ms. Pac‑Man guides: `MsPacman_GUI_Training.md` and `MsPacman_RL_Documentation.md`.
  - Example script: `scripts/train_mspacman_gui.py` for course demos.

## AI/ML algorithms implemented here and how they’re used

Primary algorithms used in this project
- PPO (Proximal Policy Optimization)
  - Where: Stable-Baselines3 PPO via RL Zoo (`rl_zoo3`), driven by `train.py`.
  - Config: `hyperparams/ppo.yml` (Atari section) for Ms. Pac‑Man; default `CnnPolicy` (NatureCNN backbone), Atari wrappers, frame stack=4.
  - Usage:
    - Headless: `python train.py --algo ppo --env MsPacmanNoFrameskip-v4 ...`
    - GUI: `--hyperparams n_envs:1 --env-kwargs render_mode:human` (see command below).
  - Enhanced mode: in `enhanced_training.py` using larger heads and reward shaping.

- A2C (Advantage Actor-Critic)
  - Where: SB3 A2C; supported by RL Zoo and `enhanced_training.py`.
  - Usage: `python train.py --algo a2c --env MsPacmanNoFrameskip-v4 ...` or `python enhanced_training.py --algorithm A2C ...`.

- DQN (Deep Q-Network)
  - Where: SB3 DQN; supported by RL Zoo and `enhanced_training.py` (with replay buffer and epsilon-greedy schedule).
  - Usage: `python train.py --algo dqn --env MsPacmanNoFrameskip-v4 ...` or `python enhanced_training.py --algorithm DQN ...`.

Additional algorithms available in this repo (via RL Zoo hyperparams)
- A2C, PPO (also `ppo_lstm.yml` for LSTM variants), DQN, QR‑DQN, TRPO, SAC, TD3, TQC, DDPG, ARS, HER (for goal-based envs). See `hyperparams/*.yml`.
- Ms. Pac‑Man is discrete-action Atari; recommended choices here are PPO, A2C, DQN (and optionally QR‑DQN).
- Invoke any supported algo with: `python train.py --algo <algo> --env MsPacmanNoFrameskip-v4 ...` (ensure a matching `<algo>.yml` exists).

Where key pieces live
- Training CLI & pipeline: `train.py` → `rl_zoo3/train.py` → `rl_zoo3/exp_manager.py` (ExperimentManager).
- Enhanced training: `enhanced_training.py` (custom wrapper, callbacks, larger nets, algorithm comparison).
- Atari defaults and schedules: `hyperparams/ppo.yml` (Atari block for Ms. Pac‑Man).
- Wrappers: `stable_baselines3.common.atari_wrappers.AtariWrapper`, `VecFrameStack(4)`, `VecTransposeImage` (applied by RL Zoo).
- Models: SB3 `CnnPolicy` (NatureCNN) with separate actor/critic heads; enhanced heads configured via `policy_kwargs`.

## Model and preprocessing used for Ms. Pac‑Man
- Observations: 84×84 grayscale frames, stacked 4 deep → transposed to channel-first (4, 84, 84).
- NatureCNN backbone (SB3 CnnPolicy):
  - Conv(32, 8×8, stride 4) → ReLU
  - Conv(64, 4×4, stride 2) → ReLU
  - Conv(64, 3×3, stride 1) → ReLU
  - Flatten → Linear(512) → ReLU
  - Heads: actor → logits(|A|), critic → scalar value
- AtariWrapper: frame max, grayscale, resize, reward clipping (sign-based by default in Atari setups).
- Enhanced nets: optional larger MLP heads (pi/vf: 512‑512‑256) via `enhanced_training.py`.

## Training/eval data flow at a glance
1) CLI parses args and loads hyperparams from `hyperparams/<algo>.yml`.
2) VecEnv is created with wrappers (AtariWrapper → FrameStack(4) → Transpose) and optional `render_mode=human` for GUI.
3) PPO/A2C/DQN collect rollouts, compute advantages (GAE for PPO/A2C), and optimize.
4) Callbacks handle evaluation, checkpointing, early stop (in enhanced script).
5) Outputs (model, configs, monitor CSVs, TB events) are saved under `logs/<algo>/<env>_<RUN_ID>/`.

<img src="images/car.jpg" align="right" width="40%"/>

RL Baselines3 Zoo is a training framework for Reinforcement Learning (RL), using [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3).

It provides scripts for training, evaluating agents, tuning hyperparameters, plotting results and recording videos.

In addition, it includes a collection of tuned hyperparameters for common environments and RL algorithms, and agents trained with those settings.


We are **looking for contributors** to complete the collection!

Goals of this repository:

1. Provide a simple interface to train and enjoy RL agents
2. Benchmark the different Reinforcement Learning algorithms
3. Provide tuned hyperparameters for each environment and RL algorithm
4. Have fun with the trained agents!

This is the SB3 version of the original SB2 [rl-zoo](https://github.com/araffin/rl-baselines-zoo).

Note: although SB3 and the RL Zoo are compatible with Numpy>=2.0, you will need Numpy<2 to run agents on pybullet envs (see [issue](https://github.com/bulletphysics/bullet3/issues/4649)).

## Documentation

Documentation is available online: [https://rl-baselines3-zoo.readthedocs.io/](https://rl-baselines3-zoo.readthedocs.io)

## Installation

### Minimal installation

From source:
```
pip install -e .
```

As a python package:
```
pip install rl_zoo3
```

Note: you can do `python -m rl_zoo3.train` from any folder and you have access to `rl_zoo3` command line interface, for instance, `rl_zoo3 train` is equivalent to `python train.py`

### Full installation (with extra envs and test dependencies)

```
apt-get install swig cmake ffmpeg
pip install -r requirements.txt
pip install -e .[plots,tests]
```

Please see [Stable Baselines3 documentation](https://stable-baselines3.readthedocs.io/en/master/) for alternatives to install stable baselines3.

## Train an Agent

The hyperparameters for each environment are defined in `hyperparameters/algo_name.yml`.

If the environment exists in this file, then you can train an agent using:
```
python train.py --algo algo_name --env env_id
```

Evaluate the agent every 10000 steps using 10 episodes for evaluation (using only one evaluation env):
```
python train.py --algo sac --env HalfCheetahBulletEnv-v0 --eval-freq 10000 --eval-episodes 10 --n-eval-envs 1
```

More examples are available in the [documentation](https://rl-baselines3-zoo.readthedocs.io).


### MsPacman training with GUI (human render)

To train PPO on MsPacman with the GUI enabled (Gym `render_mode=human`) for a short run, use:

```
python train.py --algo ppo --env MsPacmanNoFrameskip-v4 --hyperparams n_envs:1 --env-kwargs render_mode:human --eval-env-kwargs render_mode:rgb_array --n-timesteps 1024 --log-interval 1 --eval-freq -1 --save-freq -1 -P
```


## Integrations

The RL Zoo has some integration with other libraries/services like Weights & Biases for experiment tracking or Hugging Face for storing/sharing trained models. You can find out more in the [dedicated section](https://rl-baselines3-zoo.readthedocs.io/en/master/guide/integrations.html) of the documentation.

## Plot Scripts

Please see the [dedicated section](https://rl-baselines3-zoo.readthedocs.io/en/master/guide/plot.html) of the documentation.

## Enjoy a Trained Agent

**Note: to download the repo with the trained agents, you must use `git clone --recursive https://github.com/DLR-RM/rl-baselines3-zoo`** in order to clone the submodule too.


If the trained agent exists, then you can see it in action using:
```
python enjoy.py --algo algo_name --env env_id
```

For example, enjoy A2C on Breakout during 5000 timesteps:
```
python enjoy.py --algo a2c --env BreakoutNoFrameskip-v4 --folder rl-trained-agents/ -n 5000
```

## Hyperparameters Tuning

Please see the [dedicated section](https://rl-baselines3-zoo.readthedocs.io/en/master/guide/tuning.html) of the documentation.

## Custom Configuration

Please see the [dedicated section](https://rl-baselines3-zoo.readthedocs.io/en/master/guide/config.html) of the documentation.

## Current Collection: 200+ Trained Agents!

Final performance of the trained agents can be found in [`benchmark.md`](./benchmark.md). To compute them, simply run `python -m rl_zoo3.benchmark`.

List and videos of trained agents can be found on our Huggingface page: https://huggingface.co/sb3

*NOTE: this is not a quantitative benchmark as it corresponds to only one run (cf [issue #38](https://github.com/araffin/rl-baselines-zoo/issues/38)). This benchmark is meant to check algorithm (maximal) performance, find potential bugs and also allow users to have access to pretrained agents.*

### Atari Games

7 atari games from OpenAI benchmark (NoFrameskip-v4 versions).

|  RL Algo |  BeamRider         | Breakout           | Enduro             |  Pong | Qbert | Seaquest           | SpaceInvaders      |
|----------|--------------------|--------------------|--------------------|-------|-------|--------------------|--------------------|
| A2C      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| PPO      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| DQN      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| QR-DQN   | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |

Additional Atari Games (to be completed):

|  RL Algo |  MsPacman   | Asteroids | RoadRunner |
|----------|-------------|-----------|------------|
| A2C      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| PPO      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| DQN      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| QR-DQN   | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |


### Classic Control Environments

|  RL Algo |  CartPole-v1 | MountainCar-v0 | Acrobot-v1 | Pendulum-v1 | MountainCarContinuous-v0 |
|----------|--------------|----------------|------------|--------------------|--------------------------|
| ARS      | :heavy_check_mark: | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| A2C      | :heavy_check_mark: | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| PPO      | :heavy_check_mark: | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| DQN      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | N/A                | N/A |
| QR-DQN   | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | N/A                | N/A |
| DDPG     |  N/A |  N/A  | N/A | :heavy_check_mark: | :heavy_check_mark: |
| SAC      |  N/A |  N/A  | N/A | :heavy_check_mark: | :heavy_check_mark: |
| TD3      |  N/A |  N/A  | N/A | :heavy_check_mark: | :heavy_check_mark: |
| TQC      |  N/A |  N/A  | N/A | :heavy_check_mark: | :heavy_check_mark: |
| TRPO     | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |


### Box2D Environments

|  RL Algo |  BipedalWalker-v3 | LunarLander-v2 | LunarLanderContinuous-v2 |  BipedalWalkerHardcore-v3 | CarRacing-v0 |
|----------|--------------|----------------|------------|--------------|--------------------------|
| ARS      |  | :heavy_check_mark: | | :heavy_check_mark: | |
| A2C      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | |
| PPO      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | |
| DQN      | N/A | :heavy_check_mark: | N/A | N/A | N/A |
| QR-DQN   | N/A | :heavy_check_mark: | N/A | N/A | N/A |
| DDPG     | :heavy_check_mark: | N/A | :heavy_check_mark: | | |
| SAC      | :heavy_check_mark: | N/A | :heavy_check_mark: | :heavy_check_mark: | |
| TD3      | :heavy_check_mark: | N/A | :heavy_check_mark: | :heavy_check_mark: | |
| TQC      | :heavy_check_mark: | N/A | :heavy_check_mark: | :heavy_check_mark: | |
| TRPO     | | :heavy_check_mark: | :heavy_check_mark: | | |

### PyBullet Environments

See https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet/gym/pybullet_envs.
Similar to [MuJoCo Envs](https://gym.openai.com/envs/#mujoco) but with a ~free~ (MuJoCo 2.1.0+ is now free!) easy to install simulator: pybullet. We are using `BulletEnv-v0` version.

Note: those environments are derived from [Roboschool](https://github.com/openai/roboschool) and are harder than the Mujoco version (see [Pybullet issue](https://github.com/bulletphysics/bullet3/issues/1718#issuecomment-393198883))

|  RL Algo |  Walker2D | HalfCheetah | Ant | Reacher |  Hopper | Humanoid |
|----------|-----------|-------------|-----|---------|---------|----------|
| ARS      |  |  |  |  |  | |
| A2C      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | |
| PPO      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | |
| DDPG     | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | |
| SAC      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | |
| TD3      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | |
| TQC      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | |
| TRPO     | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | |

PyBullet Envs (Continued)

|  RL Algo |  Minitaur | MinitaurDuck | InvertedDoublePendulum | InvertedPendulumSwingup |
|----------|-----------|-------------|-----|---------|
| A2C      | | | | |
| PPO      | | | | |
| DDPG     | | | | |
| SAC      | | | | |
| TD3      | | | | |
| TQC      | | | | |

### MuJoCo Environments

|  RL Algo |  Walker2d | HalfCheetah | Ant | Swimmer |  Hopper | Humanoid |
|----------|-----------|-------------|-----|---------|---------|----------|
| ARS      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |  |
| A2C      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| PPO      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | |
| DDPG     |  |  |  |  |  | |
| SAC      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| TD3      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| TQC      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| TRPO      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |  |

### Robotics Environments

# DS510 Group Project — Ms. Pac‑Man Reinforcement Learning

Train, evaluate, visualize, and play a Ms. Pac‑Man agent using Stable‑Baselines3 (SB3) with the RL Baselines3 Zoo tooling. This repository adds GUI training support, enhanced training options, plotting utilities, and an interactive play‑and‑rank experience.


## What this project does

- Trains an RL agent (primarily PPO) to play Ms. Pac‑Man using Atari preprocessing and CNN policies.
- Supports both headless and GUI training (visible game window) for demos and debugging.
- Provides an enhanced training script with reward shaping, larger networks, callbacks, and algorithm comparison.
- Lets you play as a human or watch a trained bot, and records scores in a simple leaderboard.
- Includes plotting scripts to visualize training metrics and export high‑resolution figures.


## Quick start (Windows PowerShell)

1) Install dependencies (recommended Python 3.9–3.10):

```powershell
pip install -r requirements.txt
```

Optional (editable install of the local package for CLI conveniences):

```powershell
pip install -e .
```

2) Train PPO on Ms. Pac‑Man (headless, short run):

```powershell
python train.py --algo ppo --env MsPacmanNoFrameskip-v4 --n-timesteps 50000 -P
```

3) Train with a visible GUI window (single env, short smoke test):

```powershell
python train.py --algo ppo --env MsPacmanNoFrameskip-v4 --hyperparams n_envs:1 `
  --env-kwargs render_mode:human --eval-env-kwargs render_mode:rgb_array `
  --n-timesteps 1024 --log-interval 1 --eval-freq -1 --save-freq -1 -P
```

4) Watch a trained agent (if you have a saved model under logs or rl-trained-agents):

```powershell
python enjoy.py --algo ppo --env MsPacmanNoFrameskip-v4
```


## How it works (high level)

- We use Stable‑Baselines3 with the standard Atari preprocessing pipeline:
  - AtariWrapper (grayscale, resize to 84×84, frame max, reward clipping)
  - VecFrameStack(4) to approximate velocities and motion
  - Channel transpose for CNN inputs
- Policy is a CNN (NatureCNN backbone) with separate actor/critic heads (SB3 CnnPolicy).
- The RL Zoo experiment manager loads defaults from `hyperparams/ppo.yml` and builds VecEnvs, callbacks, and logging.
- Results (models, monitor CSVs, configs) are saved under `logs/<algo>/<env>_<RUN_ID>/`.


## Key features

- One‑line training via `train.py` powered by RL Zoo.
- GUI training support for demos: `--env-kwargs render_mode:human` with `--hyperparams n_envs:1`.
- Enhanced training (`enhanced_training.py`): reward shaping, bigger nets, early stopping, checkpoints, and algorithm comparison (PPO/A2C/DQN).
- Interactive leaderboard: play as human, watch bot, and store scores in `scoreboard.json`.
- Plotting utilities: export loss/metrics charts from CSV or TensorBoard.


## Installation and setup

Prerequisites
- Windows 10/11 (PowerShell examples). Linux/macOS also work with minor command changes.
- Python 3.9 or 3.10 recommended.

Steps
1) Clone this repository.
2) Install dependencies:
   - `pip install -r requirements.txt`
   - Optional: `pip install -e .` to register the local `rl_zoo3` package.
3) (Optional) For Gymnasium ALE v5 envs (`ALE/MsPacman-v5`), install Atari extras and ROMs:
   - `pip install gymnasium[atari] ale-py AutoROM AutoROM.accept-rom-license`
   - Run `AutoROM` once to install ROMs.


## Training recipes

Headless PPO (faster)

```powershell
python train.py --algo ppo --env MsPacmanNoFrameskip-v4 --n-timesteps 500000 -P
```

GUI training (visible game window)

```powershell
python train.py --algo ppo --env MsPacmanNoFrameskip-v4 --hyperparams n_envs:1 `
  --env-kwargs render_mode:human --eval-env-kwargs render_mode:rgb_array `
  --n-timesteps 1024 --log-interval 1 --eval-freq -1 --save-freq -1 -P
```

TensorBoard logging

```powershell
python train.py --algo ppo --env MsPacmanNoFrameskip-v4 --n-timesteps 100000 -tb runs -P
```

Use Gymnasium ALE v5 id (requires extras)

```powershell
python train.py --algo ppo --env ALE/MsPacman-v5 --n-timesteps 100000 -P
```

Where outputs go
- `logs/ppo/MsPacmanNoFrameskip-v4_<RUN_ID>/`
  - Final model: `MsPacmanNoFrameskip-v4.zip`
  - Config snapshots: `<env>/config.yml`, `args.yml`, `command.txt`
  - Episode logs: `*.monitor.csv`
  - TensorBoard (if `-tb` is used)


## Enhanced training (reward shaping and comparisons)

The script `enhanced_training.py` adds reward shaping, larger networks, early stopping, and periodic evaluation. You can also compare PPO, A2C, and DQN.

Train an enhanced PPO agent

```powershell
python enhanced_training.py --algorithm PPO --timesteps 500000 --n-envs 16 --eval-freq 10000 --save-path .\enhanced_models\
```

Compare algorithms quickly

```powershell
python enhanced_training.py --compare --timesteps 100000
```


## Play, watch, and rank

Use `play_and_rank.py` to play as a human (arrow keys), watch a bot, and update `scoreboard.json`.

```powershell
python play_and_rank.py
```

If needed, update the model path inside `play_and_rank.py` to point to a recent `.zip` in `logs/ppo/...`.


## Plotting and analysis

- Quick 4K loss curve export from CSV:

```powershell
python plot_pacman_loss.py -i logs\ppo\MsPacmanNoFrameskip-v4_1\MsPacmanNoFrameskip-v4 -o pacman_loss_4k.png -r 4k
```

- Additional plotting scripts live under `scripts/` (e.g., TensorBoard‑based exporters).


## Project structure (high level)

- `train.py` — Entry point calling RL Zoo’s trainer
- `enjoy.py` — Enjoy a trained model
- `enhanced_training.py` — Reward shaping, callbacks, and algorithm comparison
- `play_and_rank.py` — Human play, bot playback, and persistent scoreboard
- `plot_pacman_loss.py` — Export training loss plot (CSV → PNG)
- `hyperparams/ppo.yml` — Default PPO hyperparameters for Atari (Ms. Pac‑Man)
- `rl_zoo3/` — RL Zoo code (experiment manager, CLI, wrappers, utils)
- `logs/` — Training outputs (models, configs, CSVs, TB logs)
- `rl-trained-agents/` — Example pre‑trained agents layout
- `docs/` — Additional documentation (Sphinx) and guides


## Requirements and compatibility

- Python: 3.9 or 3.10 recommended
- Key packages: `stable-baselines3`, `gym==0.26.2`, `cloudpickle`, `moviepy`, optional `wandb`
- GPU: optional; CPU works for demos. For long runs, GPU accelerates learning.

See `requirements.txt` for the full list and versions used in this course project.


## Troubleshooting (FAQ)

- No GUI window appears
  - Ensure `--hyperparams n_envs:1` and `--env-kwargs render_mode:human`.
  - On Windows, the window may be behind your IDE; check the taskbar.

- NameError: `human` is not defined
  - Use `render_mode:human` exactly. If using a different RL Zoo variant, you may need quoting:
    `--env-kwargs 'render_mode:"human"'`.

- Multiple windows open
  - Set `--hyperparams n_envs:1`.

- Slow training with GUI
  - GUI is for demos and debugging. Prefer headless training for performance.

- `ALE/MsPacman-v5` not found
  - Install `gymnasium[atari] ale-py AutoROM AutoROM.accept-rom-license` and run `AutoROM` to install ROMs.


## Contributing (DS510)

- Fork or create a feature branch, keep changes focused, and add brief docs/tests when relevant.
- For larger changes (new scripts, wrappers), add a short note to this README and/or `docs/`.


## Acknowledgements

This project builds on the excellent RL Baselines3 Zoo and Stable‑Baselines3 projects. We adapted the training pipeline and configuration system for Ms. Pac‑Man demos and coursework.


## License

See `LICENSE` in this repository for licensing terms.
