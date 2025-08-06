#!/usr/bin/env python3
"""
Quick algorithm comparison script to demonstrate Q-Learning vs SARSA.

This script trains both algorithms and compares their performance.
"""

from apr import WarehouseEnv
from apr.agents import create_agent
from apr.train import run_episode
import matplotlib.pyplot as plt
import numpy as np


def compare_algorithms(episodes=100, max_steps=200):
    """Compare Q-Learning and SARSA performance."""
    
    print("🏁 Starting Algorithm Comparison")
    print("=" * 50)
    
    # Environment setup
    env = WarehouseEnv(seed=42)
    
    # Algorithm configurations
    algorithms = {
        'Q-Learning': {
            'name': 'q_learning',
            'params': {'alpha': 0.1, 'gamma': 0.95, 'epsilon': 0.3}
        },
        'Double Q-Learning': {
            'name': 'double_q_learning',
            'params': {'alpha': 0.1, 'gamma': 0.95, 'epsilon': 0.3}
        },
        'SARSA': {
            'name': 'sarsa',
            'params': {'alpha': 0.1, 'gamma': 0.95, 'epsilon': 0.3}
        }
    }
    
    results = {}
    
    for algo_display_name, config in algorithms.items():
        print(f"\n🤖 Training {algo_display_name}...")
        
        # Create agent
        agent = create_agent(
            config['name'],
            env.observation_space,
            env.action_space,
            **config['params']
        )
        
        # Training
        episode_rewards = []
        for episode in range(episodes):
            reward = run_episode(env, agent, training=True)
            episode_rewards.append(reward)
            
            if (episode + 1) % 20 == 0:
                avg_reward = np.mean(episode_rewards[-20:])
                print(f"  Episode {episode + 1:3d}: Avg Reward = {avg_reward:6.1f}")
        
        results[algo_display_name] = episode_rewards
        
        # Final performance
        final_avg = np.mean(episode_rewards[-20:])
        print(f"  ✅ {algo_display_name} Final Avg: {final_avg:.1f}")
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    
    # Plot learning curves
    plt.subplot(1, 2, 1)
    for algo_name, rewards in results.items():
        # Smooth with moving average
        window = 10
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards)), smoothed, label=algo_name, linewidth=2)
    
    plt.title('Learning Curves (Smoothed)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot final performance comparison
    plt.subplot(1, 2, 2)
    final_performance = {name: np.mean(rewards[-20:]) for name, rewards in results.items()}
    colors = ['skyblue', 'lightgreen', 'lightcoral']
    bars = plt.bar(final_performance.keys(), final_performance.values(), 
                   color=colors[:len(final_performance)])
    plt.title('Final Performance (Last 20 Episodes)')
    plt.ylabel('Average Reward')
    
    # Add value labels on bars
    for bar, value in zip(bars, final_performance.values()):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print("\n📊 Comparison Summary:")
    print("=" * 50)
    for algo_name, rewards in results.items():
        final_avg = np.mean(rewards[-20:])
        max_reward = np.max(rewards)
        print(f"{algo_name:10}: Final Avg = {final_avg:6.1f}, Max = {max_reward:6.1f}")
    
    return results


if __name__ == "__main__":
    results = compare_algorithms(episodes=200, max_steps=200)
    print("\n🎉 Algorithm comparison complete!")