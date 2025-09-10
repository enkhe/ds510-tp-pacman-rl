#!/usr/bin/env python3
"""
Model evaluation and comparison script for Ms. Pac-Man agents.
"""

import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt


def make_eval_env(env_id="ALE/MsPacman-v5"):
    """Create evaluation environment."""
    def _init():
        env = gym.make(env_id)
        env = AtariWrapper(env)
        env = Monitor(env)
        return env
    return _init


def evaluate_model(model, env, n_episodes=10):
    """Evaluate a trained model."""
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            done = done or truncated
            
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.1f}, Length = {episode_length}")
    
    return episode_rewards, episode_lengths


def test_baseline_model():
    """Test the baseline trained model from the logs directory."""
    try:
        # Create environment
        eval_env = DummyVecEnv([make_eval_env()])
        eval_env = VecFrameStack(eval_env, n_stack=4)
        eval_env = VecTransposeImage(eval_env)
        
        # Try to load the best model
        model_path = "logs/ppo/ALE-MsPacman-v5_1/best_model.zip"
        if os.path.exists(model_path):
            print("Loading baseline PPO model...")
            model = PPO.load(model_path, env=eval_env)
            
            print("Evaluating baseline model...")
            rewards, lengths = evaluate_model(model, eval_env, n_episodes=5)
            
            print(f"\nBaseline Model Results:")
            print(f"Average Reward: {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")
            print(f"Average Length: {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
            
            eval_env.close()
            return rewards, lengths
        else:
            print(f"Model not found at {model_path}")
            return None, None
            
    except Exception as e:
        print(f"Error testing baseline model: {e}")
        return None, None


def train_quick_comparison():
    """Train a quick comparison model with enhanced hyperparameters."""
    print("Training quick comparison model with enhanced hyperparameters...")
    
    # Create environment
    train_env = DummyVecEnv([make_eval_env()])
    train_env = VecFrameStack(train_env, n_stack=4)
    train_env = VecTransposeImage(train_env)
    
    # Enhanced hyperparameters
    model = PPO(
        'CnnPolicy',
        train_env,
        learning_rate=5e-4,
        n_steps=256,
        batch_size=256,
        n_epochs=8,
        gamma=0.99,
        gae_lambda=0.98,
        clip_range=0.1,
        ent_coef=0.005,
        vf_coef=0.75,
        max_grad_norm=0.5,
        verbose=1
    )
    
    # Train for fewer timesteps for quick comparison
    model.learn(total_timesteps=50000, progress_bar=True)
    
    # Save the model
    os.makedirs("quick_test_models", exist_ok=True)
    model.save("quick_test_models/enhanced_ppo")
    
    # Evaluate
    print("Evaluating enhanced model...")
    rewards, lengths = evaluate_model(model, train_env, n_episodes=5)
    
    print(f"\nEnhanced Model Results:")
    print(f"Average Reward: {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")
    print(f"Average Length: {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
    
    train_env.close()
    return rewards, lengths


def compare_performance():
    """Compare baseline vs enhanced performance."""
    print("="*60)
    print("Ms. Pac-Man Agent Performance Comparison")
    print("="*60)
    
    # Test baseline model
    baseline_rewards, baseline_lengths = test_baseline_model()
    
    # Train and test enhanced model
    enhanced_rewards, enhanced_lengths = train_quick_comparison()
    
    # Compare results
    if baseline_rewards and enhanced_rewards:
        print("\n" + "="*60)
        print("PERFORMANCE COMPARISON")
        print("="*60)
        
        baseline_avg = np.mean(baseline_rewards)
        enhanced_avg = np.mean(enhanced_rewards)
        improvement = ((enhanced_avg - baseline_avg) / baseline_avg) * 100
        
        print(f"Baseline Average Reward:  {baseline_avg:.1f}")
        print(f"Enhanced Average Reward:  {enhanced_avg:.1f}")
        print(f"Improvement:              {improvement:+.1f}%")
        
        # Create simple comparison plot
        try:
            plt.figure(figsize=(10, 6))
            
            plt.subplot(1, 2, 1)
            plt.bar(['Baseline', 'Enhanced'], [baseline_avg, enhanced_avg])
            plt.title('Average Reward Comparison')
            plt.ylabel('Average Reward')
            
            plt.subplot(1, 2, 2)
            plt.plot(baseline_rewards, 'o-', label='Baseline', alpha=0.7)
            plt.plot(enhanced_rewards, 's-', label='Enhanced', alpha=0.7)
            plt.title('Episode Rewards')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig('performance_comparison.png', dpi=150, bbox_inches='tight')
            print(f"\nPerformance comparison plot saved as 'performance_comparison.png'")
            
        except Exception as e:
            print(f"Could not create plot: {e}")


def main():
    """Main function."""
    compare_performance()


if __name__ == "__main__":
    main()
