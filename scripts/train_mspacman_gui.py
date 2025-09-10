import argparse
from pathlib import Path
from typing import Optional

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage

# Optional but recommended Atari preprocessing
try:
    from stable_baselines3.common.atari_wrappers import AtariWrapper
except Exception:
    AtariWrapper = None  # type: ignore


def resolve_env_id(preferred: Optional[str] = None) -> str:
    """Try common Ms. Pac-Man env IDs across Gym/Gymnasium and return a working one."""
    candidates = []
    if preferred:
        candidates.append(preferred)
    # Common IDs across versions
    candidates += [
        "MsPacmanNoFrameskip-v4",
        "ALE/MsPacman-v5",
        "MsPacman-v4",
    ]
    tried = []
    for env_id in candidates:
        if env_id in tried:
            continue
        tried.append(env_id)
        try:
            # Probe creation and immediately close
            env = gym.make(env_id, render_mode="human")
            env.close()
            print(f"Using env id: {env_id}")
            return env_id
        except Exception as e:
            print(f"Env probe failed for {env_id}: {e}")
            continue
    raise RuntimeError(
        "Could not create a Ms. Pac-Man environment. Install Atari dependencies or try another ID"
    )


def make_env(env_id: str):
    def _thunk():
        env = gym.make(env_id, render_mode="human")
        # Apply Atari preprocessing if available
        if AtariWrapper is not None:
            try:
                env = AtariWrapper(env)
            except Exception as e:
                print(f"Warning: AtariWrapper could not be applied: {e}")
        return env

    return _thunk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="auto", type=str, help="Env ID or 'auto' to try common MsPacman IDs")
    parser.add_argument("--timesteps", default=10_000, type=int)
    parser.add_argument("--save-dir", default="logs/ppo_pacman_gui", type=str)
    args = parser.parse_args()

    env_id = resolve_env_id(None if args.env == "auto" else args.env)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Single GUI env
    vec_env = DummyVecEnv([make_env(env_id)])
    # Stack 4 frames for approximate Markov property
    vec_env = VecFrameStack(vec_env, n_stack=4)
    # Ensure channel-first format for CNN policies
    vec_env = VecTransposeImage(vec_env)

    # Roughly match RL Zoo Atari PPO defaults (small run)
    model = PPO(
        policy="CnnPolicy",
        env=vec_env,
        n_steps=128,
        batch_size=256,
        n_epochs=4,
        learning_rate=2.5e-4,
        ent_coef=0.01,
        vf_coef=0.5,
        tensorboard_log=str(save_dir / "tb"),
        verbose=1,
    )

    try:
        model.learn(total_timesteps=args.timesteps, progress_bar=True)
    finally:
        # Always try to save and close even if interrupted
        try:
            model.save(str(save_dir / "MsPacmanNoFrameskip-v4"))
        except Exception as e:
            print(f"Warning: could not save model: {e}")
        try:
            vec_env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
