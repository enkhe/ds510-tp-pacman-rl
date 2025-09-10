#!/usr/bin/env python3
"""
Simple test script for the trained Ms. Pac-Man agent.
"""

import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.monitor import Monitor


def test_trained_agent():
    """Test the trained PPO agent."""
    try:
        # Load model directly
        model_path = "logs/ppo/ALE-MsPacman-v5_1/best_model.zip"
        
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}")
            return
            
        print("Loading trained PPO model...")
        model = PPO.load(model_path)
        
        # Create single environment for testing
        print("Creating test environment...")
        env = gym.make("MsPacmanNoFrameskip-v4")
        env = AtariWrapper(env)
        env = Monitor(env)
        
        # Test for 5 episodes
        print("Testing agent for 5 episodes...")
        rewards = []
        
        for episode in range(5):
            obs, _ = env.reset()
            episode_reward = 0
            steps = 0
            
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                steps += 1
                
                if done or truncated:
                    break
                    
            rewards.append(episode_reward)
            print(f"Episode {episode + 1}: Reward = {episode_reward:.0f}, Steps = {steps}")
        
        # Calculate statistics
        avg_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        
        print(f"\nResults after 5 episodes:")
        print(f"Average Reward: {avg_reward:.1f} ± {std_reward:.1f}")
        print(f"Min Reward: {min(rewards):.0f}")
        print(f"Max Reward: {max(rewards):.0f}")
        
        env.close()
        
    except Exception as e:
        print(f"Error testing agent: {e}")
        import traceback
        traceback.print_exc()


def train_improved_agent():
    """Train an improved agent with better hyperparameters."""
    print("\nTraining improved agent with enhanced hyperparameters...")
    
    try:
        # Create training environment
        def make_env():
            env = gym.make("MsPacmanNoFrameskip-v4")
            env = AtariWrapper(env)
            env = Monitor(env)
            return env
        
        # Create vectorized environment
        train_env = DummyVecEnv([make_env for _ in range(4)])  # 4 parallel environments
        train_env = VecFrameStack(train_env, n_stack=4)
        train_env = VecTransposeImage(train_env)
        
        # Create model with improved hyperparameters
        model = PPO(
            'CnnPolicy',
            train_env,
            learning_rate=2.5e-4,  # Lower learning rate
            n_steps=256,           # More steps per update
            batch_size=256,        # Larger batch size
            n_epochs=8,            # More epochs
            gamma=0.99,            # Standard discount factor
            gae_lambda=0.95,       # GAE lambda
            clip_range=0.1,        # PPO clip range
            ent_coef=0.01,         # Entropy coefficient
            vf_coef=0.5,           # Value function coefficient
            max_grad_norm=0.5,     # Gradient clipping
            verbose=1
        )
        
        # Train for more timesteps
        print("Training for 100,000 timesteps...")
        model.learn(total_timesteps=100000, progress_bar=True)
        
        # Save improved model
        os.makedirs("improved_models", exist_ok=True)
        model.save("improved_models/improved_ppo_mspacman")
        print("Improved model saved to 'improved_models/improved_ppo_mspacman.zip'")
        
        # Test improved model
        print("\nTesting improved model...")
        test_env = make_env()
        
        rewards = []
        for episode in range(3):
            obs, _ = test_env.reset()
            episode_reward = 0
            steps = 0
            
            while True:
                # Get observation for vectorized environment
                obs_vec = np.expand_dims(obs, axis=0)
                obs_vec = np.stack([obs_vec] * 4, axis=-1)  # Stack 4 frames
                obs_vec = np.transpose(obs_vec, (0, 3, 1, 2))  # Transpose for CNN
                
                action, _ = model.predict(obs_vec, deterministic=True)
                obs, reward, done, truncated, info = test_env.step(action[0])
                episode_reward += reward
                steps += 1
                
                if done or truncated:
                    break
                    
            rewards.append(episode_reward)
            print(f"Episode {episode + 1}: Reward = {episode_reward:.0f}, Steps = {steps}")
        
        print(f"\nImproved model average reward: {np.mean(rewards):.1f}")
        
        train_env.close()
        test_env.close()
        
    except Exception as e:
        print(f"Error training improved agent: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function."""
    print("="*60)
    print("Ms. Pac-Man Agent Testing")
    print("="*60)
    
    # Test existing trained agent
    test_trained_agent()
    
    # Train and test improved agent
    train_improved_agent()


if __name__ == "__main__":
    main()
