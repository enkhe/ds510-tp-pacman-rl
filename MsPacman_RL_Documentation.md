# Ms. Pac-Man Reinforcement Learning Project – Comprehensive Guide

This document explains how this project trains, evaluates, visualizes, and plays a Ms. Pac-Man agent using Stable-Baselines3 and RL Zoo.

## What’s included

- Training (RL Zoo CLI): `train.py` (wraps `rl_zoo3/train.py`)
- Atari hyperparams: `hyperparams/ppo.yml` (uses AtariWrapper + frame stacking)
- Enhanced training with reward shaping: `enhanced_training.py`
- Quick demo trainer: `train_pacman.py` (PPO + VecFrameStack, short run)
- Loss plotting from CSV: `plot_pacman_loss.py`
- Interactive play + leaderboard: `play_and_rank.py`

## Environment options

- Gym Atari v4 id: `MsPacmanNoFrameskip-v4`
- ALE v5 id: `ALE/MsPacman-v5`
- Wrappers (Atari): `stable_baselines3.common.atari_wrappers.AtariWrapper` and frame stacking (4).

## Markov Decision Process (MDP) in this project

We model the game as an MDP defined by the tuple (S, A, P, R, γ):

- States S: Preprocessed Atari frames from Ms. Pac-Man. With the Atari wrapper plus VecFrameStack, the observation is a stack of the last 4 frames to approximate the Markov property for velocities and ghost motion. Images are transposed to channel-first for CNNs.
- Actions A: Discrete action set provided by ALE for Ms. Pac-Man (the minimal action set; typically movement directions and no-op). RL Zoo uses the environment’s default discrete actions.
- Transition P: Unknown, stochastic game dynamics (ghost behavior, fruit spawns). The env provides next observation and done flags; we do not learn a model (model-free).
- Reward R: By default in Atari training, rewards are clipped (sign-based) via the wrapper. In our enhanced setup we optionally shape rewards with survival bonus, score deltas, and life-loss penalties (see “Rewards and penalties”).
- Discount γ: The PPO default from Stable-Baselines3 is used unless overridden (commonly 0.99); you can set `gamma` in `hyperparams/ppo.yml` if desired.

Episodes terminate when a life is lost (with EpisodicLife-like behavior), the level/game ends, or a time limit triggers. Because raw single frames are partially observable, stacking 4 frames is used to better satisfy the MDP assumption; strictly speaking, Atari is closer to a POMDP without stacking.

Objective and learning:
- PPO optimizes the expected discounted return E[∑ γ^t r_t] with a clipped policy-gradient surrogate.
- A learned value function V(s) and GAE(λ) (default λ≈0.95 unless overridden) provide low-variance advantage estimates.
- Policy: `CnnPolicy` over stacked frames (see Atari defaults in `hyperparams/ppo.yml`).

## Rewards and penalties

Two common configurations in this repo:

1) RL Zoo defaults (via `train.py` + AtariWrapper)
- Reward clipping enabled by default
- +1 for any positive in-game reward, 0 when neutral, -1 for negative (rare in Ms. Pac-Man)
- No explicit extra penalty on life loss (life loss usually ends episode via EpisodicLife wrapper)

2) Enhanced shaping (via `enhanced_training.py` -> `EnhancedMsPacManWrapper`)
- Effective reward per step:
  - env_reward (may be clipped)
  - + (score_delta / 100.0)
  - + 0.01 survival bonus
  - -10.0 penalty if a life was lost on that step

## Training with RL Zoo (headless)

- Minimal example (50k steps):
```powershell
python train.py --algo ppo --env MsPacmanNoFrameskip-v4 --n-timesteps 50000
```
- TensorBoard logs:
```powershell
python train.py --algo ppo --env MsPacmanNoFrameskip-v4 --n-timesteps 50000 -tb runs
```
- Single env (useful for GUI later):
```powershell
python train.py --algo ppo --env MsPacmanNoFrameskip-v4 --n-timesteps 50000 --hyperparams n_envs:1
```

Artifacts are saved under `logs/ppo/<ENV>_<RUN_ID>/` (models, monitor CSVs, config, command). The final model is `<ENV>.zip` in that folder.

## Training with GUI (visible game window)

RL Zoo supports passing env kwargs; for Windows PowerShell use quoting so the value is a Python string for argparse’s StoreDict/eval:

```powershell
python train.py --algo ppo --env MsPacmanNoFrameskip-v4 --hyperparams n_envs:1 --env-kwargs 'render_mode:"human"' --eval-env-kwargs 'render_mode:"rgb_array"' --n-timesteps 50000 --log-interval 1 -P
```
Notes:
- `--hyperparams n_envs:1` prevents multiple windows (one env only).
- `--eval-env-kwargs` remains headless for faster evaluation.
- If you see: `NameError: name 'human' is not defined`, your quoting didn’t reach Python with quotes. Keep the inner double quotes as shown.

## Enhanced training (reward shaping, larger policy)

Use `enhanced_training.py` to try PPO/A2C/DQN with reward shaping and callbacks:
```powershell
python enhanced_training.py --algorithm PPO --timesteps 500000 --n-envs 16 --eval-freq 10000 --save-path .\enhanced_models\
```
- Models, checkpoints, and TensorBoard logs saved under `.\enhanced_models\`
- Wrapper adds survival bonus, score-delta bonus, and life-loss penalty.

## Quick demo trainer (short run)

`train_pacman.py` runs a short PPO session and writes logs to `logs/ppo_pacman`:
```powershell
python train_pacman.py
```
- Model saved to `logs/ppo_pacman/pacman_model.zip`

## Logs and where to find them

- RL Zoo training run: `logs/ppo/<ENV>_<RUN_ID>/`
  - Model: `<ENV>.zip` (final) and optionally checkpoints
  - Config: `<ENV>/config.yml`, `<ENV>/args.yml`, and `command.txt`
  - Episode logs: `*.monitor.csv`
  - Progress CSV: `progress.csv` (algorithm metrics; may depend on logger setup)
  - TensorBoard: if `-tb`/`--tensorboard-log` provided

- Demo trainer run: `logs/ppo_pacman/` (model + TB events)

## Plotting training loss (4K PNG)

Use `plot_pacman_loss.py` to read `progress.csv` and export a labeled 4K graph:
```powershell
python plot_pacman_loss.py -i logs\ppo\MsPacmanNoFrameskip-v4_1\MsPacmanNoFrameskip-v4 -o pacman_loss_4k.png -r 4k
```
If `progress.csv` is missing, you can alternatively use the TensorBoard event files with the existing `scripts/plot_individual_metrics.py` to export per-metric PNGs from TB logs.

## Playing the game and leaderboard

`play_and_rank.py` provides a menu to:
- Play as human (arrow keys) and save your name+score to `scoreboard.json`
- Watch a trained bot play and save its score
- View a ranked scoreboard (top 10)

Run:
```powershell
python play_and_rank.py
```
Bot model path default:
- `rl-trained-agents/a2c/MsPacmanNoFrameskip-v4_1/MsPacmanNoFrameskip-v4.zip`
Update `MODEL_PATH` in `play_and_rank.py` if your model is elsewhere (e.g., from a fresh training run in `logs/ppo/...`).

## Tips for better learning

- Train longer (millions of steps) and with multiple envs for better exploration
- Consider un-clipping rewards for score-proportional learning (disable reward clipping in the Atari wrapper)
- Use curriculum or enhanced shaping (see `enhanced_training.py`)
- Monitor with TensorBoard for PPO diagnostics (KL, clip fraction, entropy, value loss)

## Troubleshooting

- Quoting errors for GUI:
  - Error: `NameError: name 'human' is not defined`
  - Fix: ensure the value reaches Python with quotes:
    `--env-kwargs 'render_mode:"human"'`
- Missing model when launching bot in `play_and_rank.py`:
  - Update `MODEL_PATH` to an existing `.zip` model
- `progress.csv` missing:
  - Use the TB-based plotting script (`scripts/plot_individual_metrics.py`) or ensure the SB3 CSV logger is active
- Slow training with GUI:
  - Use GUI only for short demonstrations. Prefer headless for longer runs.

## Key files at a glance

- `train.py`: RL Zoo entry point for training
- `hyperparams/ppo.yml`: Atari section governs Ms. Pac-Man defaults
- `enhanced_training.py`: Reward shaping and advanced callbacks
- `train_pacman.py`: Short, self-contained PPO demo
- `plot_pacman_loss.py`: Exports 4K loss plot from `progress.csv`
- `play_and_rank.py`: Human vs. AI play and persistent scoreboard

---
If you want this guide embedded in the docs site, we can move it under `docs/` and wire it into the Sphinx navigation.
