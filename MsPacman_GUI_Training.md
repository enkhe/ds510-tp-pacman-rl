# Ms. Pac-Man GUI Training — Quick Guide

This guide explains how to launch Ms. Pac-Man training with a visible game window using the RL Zoo CLI in this repo.

## Tech stack

- Language/runtime: Python 3.10
- RL library: Stable-Baselines3 (PPO)
- Orchestration: RL Baselines3 Zoo (this repo)
- Environments: Gym/Gymnasium Atari (MsPacmanNoFrameskip-v4); ALE (optional)
- Wrappers: AtariWrapper, VecFrameStack(4), VecTransposeImage
- Deep learning: PyTorch (via SB3)
- OS: Windows (PowerShell examples)
- Logging: SB3 logger (stdout), Monitor CSV, optional TensorBoard

## Why train with a GUI?

- Demonstration and teaching: visualize agent behavior while learning.
- Debugging: verify render pipeline and wrappers.
- Short smoke tests: validate logging, paths, and configs.

Use GUI for short runs; prefer headless for long training (faster, fewer side effects).

## The command (Windows PowerShell)

```powershell
python train.py --algo ppo --env MsPacmanNoFrameskip-v4 --hyperparams n_envs:1 `
  --env-kwargs render_mode:human --eval-env-kwargs render_mode:rgb_array `
  --n-timesteps 1024 --log-interval 1 --eval-freq -1 --save-freq -1 -P
```

- Opens one Ms. Pac-Man window and trains PPO for ~1k steps (~45–60s on CPU).
- Produces a new run folder under `logs/ppo/`.

### Flags explained

- `--algo ppo`: Use PPO with `CnnPolicy` (set by Atari defaults).
- `--env MsPacmanNoFrameskip-v4`: Atari v4 Ms. Pac-Man environment id.
- `--hyperparams n_envs:1`: Single environment to avoid multiple GUI windows.
- `--env-kwargs render_mode:human`: Enable GUI rendering for training env.
- `--eval-env-kwargs render_mode:rgb_array`: Keep evaluation headless (faster, optional when eval is enabled).
- `--n-timesteps 1024`: Short run for a quick demo; increase for longer tests.
- `--log-interval 1`: Log PPO stats each iteration (every 128 steps for Atari defaults).
- `--eval-freq -1`: Disable periodic evaluation to keep the run simple.
- `--save-freq -1`: Disable checkpoint saves (final save still occurs at the end of training).
- `-P`: Show a progress bar.

Notes
- In this repo, `StoreDict` accepts unquoted values (e.g., `render_mode:human`). Quoted values also work.
- For Gymnasium ALE v5, install atari extras and ROMs and use `--env ALE/MsPacman-v5`.

## What happens under the hood

- RL Zoo loads Atari defaults from `hyperparams/ppo.yml`:
  - AtariWrapper (preprocessing + reward clipping), `frame_stack: 4`.
  - PPO hyperparams: `n_steps=128`, `batch_size=256`, `n_epochs=4`, `learning_rate≈2.5e-4`, `ent_coef=0.01`, `vf_coef=0.5`.
- A single env is created with `render_mode='human'`, wrapped, and transposed to channel-first.
- PPO trains for the specified timesteps, logging metrics each iteration.
- Model is saved in the run folder at the end.

## Neural networks used (PPO CnnPolicy)

Input and preprocessing
- Observation: 84×84 grayscale frames, stacked 4 deep (VecFrameStack) → shape (84, 84, 4), transposed to channel-first (4, 84, 84).
- AtariWrapper: max-pooling over frames, grayscale, resize, reward clipping.

Backbone (NatureCNN, shared torso)
- Conv2D(32, kernel=8×8, stride=4) → ReLU
- Conv2D(64, kernel=4×4, stride=2) → ReLU
- Conv2D(64, kernel=3×3, stride=1) → ReLU
- Flatten → Linear(512) → ReLU

Heads
- Policy (actor) head: Linear(512 → |A|) outputs logits over discrete actions.
- Value (critic) head: Linear(512 → 1) estimates V(s).

Optimization (defaults from Atari PPO)
- Algorithm: PPO with GAE(λ≈0.95), γ≈0.99, entropy bonus (ent_coef=0.01), value loss coef (vf_coef=0.5).
- Rollout: n_steps=128 per iteration, batch_size=256, n_epochs=4, learning_rate≈2.5e-4 (linear schedule).
- Objective: clipped surrogate for policy, value-function loss, and entropy regularization.

## Where outputs go

- Run folder: `logs/ppo/MsPacmanNoFrameskip-v4_<RUN_ID>/`
  - Final model: `MsPacmanNoFrameskip-v4.zip`
  - Config snapshot: `MsPacmanNoFrameskip-v4/config.yml`, `args.yml`, `command.txt`
  - Episode logs: `*.monitor.csv`
  - Eval logs: only if `--eval-freq > 0`
  - TensorBoard: only if you pass `-tb runs`

## When to use which env id

- `MsPacmanNoFrameskip-v4` (default): Works out of the box with Gym in this repo.
- `ALE/MsPacman-v5` (Gymnasium): Requires `gymnasium[atari]`, `ale-py`, and Atari ROMs (AutoROM). Use if you want Gymnasium’s v5 API.

## Troubleshooting

- No window appears:
  - Ensure you’re on Windows (not WSL/headless), and `n_envs:1`.
  - Some IDE terminals may hide the window behind the editor—check the taskbar.
- NameError: `human` is not defined:
  - Use `render_mode:human` exactly (unquoted works here); or `--env-kwargs 'render_mode:"human"'` if using a vanilla RL Zoo without our StoreDict patch.
- Multiple windows open:
  - Set `--hyperparams n_envs:1`.
- Slow training:
  - GUI is slower by design. Use GUI for short demos; train headless for performance.
- ALE/MsPacman-v5 not found:
  - Install: `pip install gymnasium[atari] ale-py AutoROM AutoROM.accept-rom-license` and run AutoROM to install ROMs.

## Next steps

- Longer run with progress and TensorBoard:
```powershell
python train.py --algo ppo --env MsPacmanNoFrameskip-v4 --hyperparams n_envs:1 `
  --env-kwargs render_mode:human --n-timesteps 50000 -tb runs -P
```
- Plot training curves from logs (see `plot_pacman_loss.py` or TB-based scripts under `scripts/`).
- Play or watch the bot with the interactive `play_and_rank.py` script.

---
For conceptual background (MDP, reward clipping vs shaping), see `MsPacman_RL_Documentation.md`.

## Example training scores (per iteration)

Below is an example slice of PPO logs during a short GUI run (~1k steps). The `ep_rew_mean` is the average episode reward over the last rollout; `iterations` increments every 128 steps (default Atari `n_steps`).

```
-------------------------------------------
| rollout/                |               |
|    ep_len_mean          | 2.01e+03      |
|    ep_rew_mean          | 200           |
| time/                   |               |
|    fps                  | 22            |
|    iterations           | 4             |
|    time_elapsed         | 22            |
|    total_timesteps      | 512           |
| train/                  |               |
|    approx_kl            | 0.00023097545 |
|    clip_fraction        | 0             |
|    clip_range           | 0.0625        |
|    entropy_loss         | -2.19         |
|    explained_variance   | 0.0694        |
|    learning_rate        | 0.000156      |
|    loss                 | 0.367         |
|    n_updates            | 12            |
|    policy_gradient_loss | -0.00209      |
|    value_loss           | 1.03          |
-------------------------------------------
```

From the monitor CSVs for recent runs:

- `logs/ppo/MsPacmanNoFrameskip-v4_4/0.monitor.csv`
  - 270.0 points, 1,945 frames
  - 170.0 points, 1,617 frames
- `logs/ppo/MsPacmanNoFrameskip-v4_6/0.monitor.csv`
  - 200.0 points, 2,009 frames

These per-episode scores reflect the clipped-reward Atari setup; raw in-game score trends are still meaningful as you scale training up.

## What these metrics mean

- rollout/
  - ep_len_mean: Average episode length (steps) over the last rollout batch.
  - ep_rew_mean: Mean episode reward over the last rollout. With Atari wrappers, rewards are typically sign-clipped.

- time/
  - fps: Effective training speed (env steps per second, including learning overhead).
  - iterations: Number of PPO iterations completed; each iteration collects n_steps per env (Atari default: 128).
  - time_elapsed: Wall-clock seconds since the run started.
  - total_timesteps: Total environment steps sampled so far (iterations × n_steps × n_envs).

- train/
  - approx_kl: Approximate KL divergence between old and new policy; monitors update size (too high can indicate instability).
  - clip_fraction: Fraction of policy ratios that hit the PPO clip boundary; gauges how often updates are being clipped.
  - clip_range: Current PPO clipping epsilon (may be scheduled/decayed).
  - entropy_loss: Negative policy entropy term; more negative implies higher entropy (more exploration).
  - explained_variance: How well the value function explains returns (1.0 is perfect, ~0 poor, <0 worse than predicting a constant).
  - learning_rate: Current learning rate (after any schedule).
  - loss: Combined training loss (policy + value + entropy terms); scale is setup-dependent.
  - n_updates: Cumulative optimizer updates performed so far across iterations.
  - policy_gradient_loss: Policy (actor) objective; typically negative when improving (due to sign convention).
  - value_loss: Value-function (critic) loss, usually mean-squared error of value prediction.

  ## Project architecture (high level)

  Main components
  - `train.py` (root): thin wrapper that calls `rl_zoo3/train.py:train()`.
  - `rl_zoo3/train.py`: parses CLI args, validates env id, sets random seeds, and builds an `ExperimentManager`.
  - `rl_zoo3/exp_manager.py` (ExperimentManager):
    - Loads hyperparams from `hyperparams/ppo.yml` (Atari section for Ms. Pac-Man).
    - Creates training/eval VecEnvs via `make_vec_env`, applying wrappers (AtariWrapper, VecFrameStack, VecTransposeImage).
    - Constructs the SB3 model (PPO) with the processed hyperparams.
    - Orchestrates learning loop, callbacks (e.g., EvalCallback, checkpoints), and saving.
  - `rl_zoo3/utils.py`: helpers for parsing dict-like CLI overrides (StoreDict), loading callbacks/wrappers, and resolving log/model paths.
  - `hyperparams/ppo.yml`: declarative defaults for Atari PPO (policy, n_steps, frame_stack, schedulers, etc.).

  Data flow
  1) CLI → parse args and hyperparams; pick env id and wrappers.
  2) Build VecEnv (n_envs=1 for GUI), apply Atari preprocessing + 4-frame stack, transpose images.
  3) For each PPO iteration: collect 128 steps → compute advantages with GAE → optimize policy/value for 4 epochs on 256-size minibatches.
  4) Log metrics (console/TB), save final model and config snapshot under `logs/ppo/...`.