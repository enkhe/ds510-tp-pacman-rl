import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import os

# --- Constants ---
ENV_NAME = "MsPacmanNoFrameskip-v4"
LOG_DIR = "logs/ppo_pacman"
MODEL_ALGO = "PPO"
TOTAL_TIMESTEPS = 25000 # A short training session for demonstration

# --- Create Environment ---
# Create the base environment
# The make_atari_env helper handles the AtariPreprocessing wrapper
env = make_atari_env(ENV_NAME, n_envs=1, seed=42)
# Stack 4 frames, a common practice for Atari games
env = VecFrameStack(env, n_stacks=4)

# --- Create Model ---
# We will use the PPO algorithm with parameters suited for Atari games
model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    tensorboard_log=LOG_DIR,
    learning_rate=2.5e-4,
    n_steps=128,
    batch_size=256,
    n_epochs=4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.1,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
)

# --- Train Model ---
print(f"Starting training for {TOTAL_TIMESTEPS} timesteps...")
# The `learn` method will save logs to the specified `tensorboard_log` directory
# It also automatically creates a `progress.csv` file inside the log folder.
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    progress_bar=True
)

# --- Save Model ---
model_path = os.path.join(LOG_DIR, "pacman_model.zip")
model.save(model_path)

print(f"Training complete. Model saved to {model_path}")
env.close()
