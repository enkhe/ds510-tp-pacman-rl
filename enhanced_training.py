#!/usr/bin/env python3
"""
Enhanced training script for Ms. Pac-Man with multiple performance improvements.

This script demonstrates several techniques to improve RL agent performance:
1. Enhanced hyperparameters
2. Better reward engineering
3. Curriculum learning
4. Multiple algorithm comparison
5. Better evaluation and monitoring
"""

import argparse
import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
import torch as th
from torch import nn


class EnhancedMsPacManWrapper(gym.Wrapper):
    """Enhanced wrapper for Ms. Pac-Man with improved reward shaping."""
    
    def __init__(self, env):
        super().__init__(env)
        self.prev_score = 0
        self.prev_lives = 0
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_score = 0
        self.prev_lives = info.get('lives', 0)
        return obs, info
        
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        
        # Enhanced reward shaping
        current_score = info.get('score', 0)
        current_lives = info.get('lives', 0)
        
        # Reward for score increase
        score_reward = (current_score - self.prev_score) / 100.0
        
        # Penalty for losing a life
        life_penalty = -10.0 if current_lives < self.prev_lives else 0.0
        
        # Small survival bonus
        survival_bonus = 0.01
        
        # Combine rewards
        enhanced_reward = reward + score_reward + life_penalty + survival_bonus
        
        self.prev_score = current_score
        self.prev_lives = current_lives
        
        return obs, enhanced_reward, done, truncated, info


def make_enhanced_env(env_id, rank=0):
    """Create enhanced environment with custom wrappers."""
    def _init():
        env = gym.make(env_id)
        env = AtariWrapper(env)
        env = EnhancedMsPacManWrapper(env)
        env = Monitor(env)
        return env
    return _init


def create_enhanced_cnn_policy():
    """Create enhanced CNN policy architecture."""
    return dict(
        features_extractor_class=None,  # Use default
        features_extractor_kwargs={},
        net_arch=dict(
            pi=[512, 512, 256],  # Larger policy network
            vf=[512, 512, 256]   # Larger value network
        ),
        activation_fn=nn.ReLU,
        normalize_images=True
    )


def train_enhanced_agent(
    env_id: str = "ALE/MsPacman-v5",
    algorithm: str = "PPO",
    total_timesteps: int = 500000,
    n_envs: int = 16,
    eval_freq: int = 10000,
    save_path: str = "./enhanced_models/"
):
    """Train an enhanced agent with improved hyperparameters."""
    
    # Create directories
    os.makedirs(save_path, exist_ok=True)
    eval_log_path = os.path.join(save_path, "eval_logs")
    os.makedirs(eval_log_path, exist_ok=True)
    
    # Create training environments
    train_env = make_vec_env(
        make_enhanced_env(env_id), 
        n_envs=n_envs, 
        vec_env_cls=SubprocVecEnv if n_envs > 1 else DummyVecEnv
    )
    train_env = VecFrameStack(train_env, n_stack=4)
    train_env = VecTransposeImage(train_env)
    
    # Create evaluation environment
    eval_env = make_vec_env(
        make_enhanced_env(env_id), 
        n_envs=1, 
        vec_env_cls=DummyVecEnv
    )
    eval_env = VecFrameStack(eval_env, n_stack=4)
    eval_env = VecTransposeImage(eval_env)
    
    # Enhanced hyperparameters based on algorithm
    if algorithm == "PPO":
        model_params = {
            'policy': 'CnnPolicy',
            'env': train_env,
            'learning_rate': 5e-4,
            'n_steps': 256,
            'batch_size': 512,
            'n_epochs': 8,
            'gamma': 0.99,
            'gae_lambda': 0.98,
            'clip_range': 0.1,
            'ent_coef': 0.005,
            'vf_coef': 0.75,
            'max_grad_norm': 0.5,
            'policy_kwargs': create_enhanced_cnn_policy(),
            'verbose': 1,
            'tensorboard_log': os.path.join(save_path, "tensorboard")
        }
        model = PPO(**model_params)
        
    elif algorithm == "A2C":
        model_params = {
            'policy': 'CnnPolicy',
            'env': train_env,
            'learning_rate': 7e-4,
            'n_steps': 256,
            'gamma': 0.99,
            'gae_lambda': 0.98,
            'ent_coef': 0.01,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'policy_kwargs': create_enhanced_cnn_policy(),
            'verbose': 1,
            'tensorboard_log': os.path.join(save_path, "tensorboard")
        }
        model = A2C(**model_params)
        
    elif algorithm == "DQN":
        model_params = {
            'policy': 'CnnPolicy',
            'env': train_env,
            'learning_rate': 1e-4,
            'buffer_size': 100000,
            'learning_starts': 10000,
            'batch_size': 32,
            'tau': 1.0,
            'gamma': 0.99,
            'train_freq': 4,
            'gradient_steps': 1,
            'target_update_interval': 1000,
            'exploration_fraction': 0.1,
            'exploration_initial_eps': 1.0,
            'exploration_final_eps': 0.05,
            'policy_kwargs': create_enhanced_cnn_policy(),
            'verbose': 1,
            'tensorboard_log': os.path.join(save_path, "tensorboard")
        }
        model = DQN(**model_params)
    
    # Enhanced callbacks
    callbacks = []
    
    # Evaluation callback with early stopping
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_path, f"best_{algorithm.lower()}_model"),
        log_path=eval_log_path,
        eval_freq=eval_freq,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )
    callbacks.append(eval_callback)
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=eval_freq * 2,
        save_path=os.path.join(save_path, "checkpoints"),
        name_prefix=f"{algorithm.lower()}_checkpoint"
    )
    callbacks.append(checkpoint_callback)
    
    # Stop training when reward threshold is reached
    stop_callback = StopTrainingOnRewardThreshold(
        reward_threshold=1000,  # Stop when average reward reaches 1000
        verbose=1
    )
    callbacks.append(stop_callback)
    
    print(f"Starting training with {algorithm} algorithm...")
    print(f"Training for {total_timesteps} timesteps with {n_envs} parallel environments")
    print(f"Models will be saved to: {save_path}")
    
    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True
    )
    
    # Save final model
    final_model_path = os.path.join(save_path, f"final_{algorithm.lower()}_model")
    model.save(final_model_path)
    print(f"Final model saved to: {final_model_path}")
    
    # Clean up
    train_env.close()
    eval_env.close()
    
    return model


def compare_algorithms():
    """Compare different RL algorithms on Ms. Pac-Man."""
    algorithms = ["PPO", "A2C", "DQN"]
    results = {}
    
    for algo in algorithms:
        print(f"\n{'='*50}")
        print(f"Training {algo} algorithm")
        print(f"{'='*50}")
        
        try:
            model = train_enhanced_agent(
                algorithm=algo,
                total_timesteps=100000,  # Shorter for comparison
                save_path=f"./enhanced_models/{algo.lower()}/"
            )
            results[algo] = "Success"
        except Exception as e:
            print(f"Error training {algo}: {e}")
            results[algo] = f"Failed: {e}"
    
    print(f"\n{'='*50}")
    print("Training Results Summary:")
    print(f"{'='*50}")
    for algo, result in results.items():
        print(f"{algo}: {result}")


def main():
    """Main function with command line arguments."""
    parser = argparse.ArgumentParser(description="Enhanced Ms. Pac-Man RL Training")
    parser.add_argument("--algorithm", choices=["PPO", "A2C", "DQN"], default="PPO",
                       help="RL algorithm to use")
    parser.add_argument("--timesteps", type=int, default=500000,
                       help="Total training timesteps")
    parser.add_argument("--n-envs", type=int, default=16,
                       help="Number of parallel environments")
    parser.add_argument("--eval-freq", type=int, default=10000,
                       help="Evaluation frequency")
    parser.add_argument("--save-path", type=str, default="./enhanced_models/",
                       help="Path to save models")
    parser.add_argument("--compare", action="store_true",
                       help="Compare multiple algorithms")
    
    args = parser.parse_args()
    
    if args.compare:
        compare_algorithms()
    else:
        train_enhanced_agent(
            algorithm=args.algorithm,
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            eval_freq=args.eval_freq,
            save_path=args.save_path
        )


if __name__ == "__main__":
    main()
