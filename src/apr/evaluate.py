"""
Evaluation module for trained reinforcement learning agents.

This module provides comprehensive evaluation capabilities for trained agents,
including performance testing, visualization, and comparison tools.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from collections import defaultdict
import pandas as pd

from .env import WarehouseEnv
from .agents.base import BaseAgent
from .agents import create_agent
from .train import run_episode


class AgentEvaluator:
    """
    Comprehensive evaluation system for trained RL agents.
    
    Provides tools for:
    - Loading and testing trained agents
    - Performance evaluation across multiple scenarios
    - Comparative analysis between agents
    - Visualization of agent behavior and performance
    """
    
    def __init__(self, env: Optional[WarehouseEnv] = None, verbose: bool = True):
        """
        Initialize the evaluator.
        
        Args:
            env: Environment to use for evaluation (creates default if None)
            verbose: Whether to print detailed information
        """
        self.env = env if env is not None else WarehouseEnv()
        self.verbose = verbose
        self.evaluation_results = {}
        
    def print_status(self, message: str) -> None:
        """Print status message if verbose mode is enabled."""
        if self.verbose:
            print(message)
    
    def load_agent(self, agent_path: Union[str, Path], algorithm: str) -> BaseAgent:
        """
        Load a trained agent from disk.
        
        Args:
            agent_path: Path to the saved agent file
            algorithm: Algorithm type (q_learning, double_q_learning, sarsa, etc.)
            
        Returns:
            Loaded agent instance
        """
        agent_path = Path(agent_path)
        
        if not agent_path.exists():
            raise FileNotFoundError(f"Agent file not found: {agent_path}")
        
        self.print_status(f"Loading {algorithm} agent from {agent_path}")
        
        # Create agent instance and load from file
        if algorithm.lower() in ['q_learning', 'qlearning']:
            from .agents.q_learning import QLearningAgent
            agent = QLearningAgent.load(agent_path)
        elif algorithm.lower() in ['double_q_learning', 'double_qlearning']:
            from .agents.double_q_learning import DoubleQLearningAgent
            agent = DoubleQLearningAgent.load(agent_path)
        elif algorithm.lower() in ['sarsa']:
            from .agents.sarsa import SarsaAgent
            agent = SarsaAgent.load(agent_path)
        else:
            raise ValueError(f"Unsupported algorithm for loading: {algorithm}")
        
        self.print_status(f"Successfully loaded agent: {agent}")
        return agent
    
    def evaluate_agent(self, 
                      agent: BaseAgent, 
                      num_episodes: int = 100,
                      seeds: Optional[List[int]] = None,
                      render: bool = False) -> Dict[str, Any]:
        """
        Evaluate agent performance across multiple episodes and scenarios.
        
        Args:
            agent: Agent to evaluate
            num_episodes: Number of episodes to run
            seeds: List of random seeds for different scenarios (None for random)
            render: Whether to render episodes (useful for debugging)
            
        Returns:
            Comprehensive evaluation results
        """
        self.print_status(f"Evaluating agent over {num_episodes} episodes")
        
        if seeds is None:
            seeds = [42]  # Default single seed
        
        all_results = {}
        
        for seed_idx, seed in enumerate(seeds):
            self.print_status(f"  Testing with seed {seed} ({seed_idx + 1}/{len(seeds)})")
            
            # Set environment seed
            test_env = WarehouseEnv(seed=seed)
            
            episode_rewards = []
            episode_lengths = []
            episode_outcomes = []
            state_visits = defaultdict(int)
            action_counts = defaultdict(lambda: defaultdict(int))
            
            for episode in range(num_episodes):
                # Run episode
                test_env.reset()
                total_reward = 0
                steps = 0
                done = False
                
                # Track trajectory
                trajectory = []
                
                while not done and steps < test_env.max_steps:
                    state = test_env.agent_pos
                    action = agent.act(state, training=False)  # No exploration during evaluation
                    
                    # Track state and action
                    state_visits[state] += 1
                    action_counts[state][action] += 1
                    trajectory.append((state, action))
                    
                    # Take step
                    next_state, reward, done, info = test_env.step(action)
                    total_reward += reward
                    steps += 1
                    
                    if render and episode == 0:  # Only render first episode
                        test_env.render()
                
                episode_rewards.append(total_reward)
                episode_lengths.append(steps)
                
                # Determine outcome
                if done and test_env.num_picked_up_items > 0 and not test_env.carrying_packages:
                    episode_outcomes.append('success')
                elif steps >= test_env.max_steps:
                    episode_outcomes.append('timeout')
                else:
                    episode_outcomes.append('failed')
                
                if (episode + 1) % 20 == 0:
                    avg_reward = np.mean(episode_rewards[-20:])
                    self.print_status(f"    Episode {episode + 1}: Avg Reward = {avg_reward:.1f}")
            
            # Calculate statistics for this seed
            seed_results = {
                'seed': seed,
                'rewards': episode_rewards,
                'lengths': episode_lengths,
                'outcomes': episode_outcomes,
                'statistics': {
                    'mean_reward': np.mean(episode_rewards),
                    'std_reward': np.std(episode_rewards),
                    'min_reward': np.min(episode_rewards),
                    'max_reward': np.max(episode_rewards),
                    'mean_length': np.mean(episode_lengths),
                    'std_length': np.std(episode_lengths),
                    'success_rate': episode_outcomes.count('success') / len(episode_outcomes),
                    'timeout_rate': episode_outcomes.count('timeout') / len(episode_outcomes),
                    'failure_rate': episode_outcomes.count('failed') / len(episode_outcomes)
                },
                'exploration_stats': {
                    'states_visited': len(state_visits),
                    'total_state_visits': sum(state_visits.values()),
                    'state_coverage': len(state_visits) / (test_env.n_rows * test_env.n_cols),
                    'action_entropy': self._calculate_action_entropy(action_counts)
                }
            }
            
            all_results[f'seed_{seed}'] = seed_results
        
        # Aggregate results across seeds
        aggregated_results = self._aggregate_seed_results(all_results)
        
        return {
            'per_seed_results': all_results,
            'aggregated_results': aggregated_results,
            'evaluation_config': {
                'num_episodes': num_episodes,
                'seeds': seeds,
                'agent_type': type(agent).__name__
            }
        }
    
    def compare_agents(self, 
                      agents: Dict[str, BaseAgent], 
                      num_episodes: int = 100,
                      seeds: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Compare multiple agents on the same evaluation scenarios.
        
        Args:
            agents: Dictionary mapping agent names to agent instances
            num_episodes: Number of episodes per agent
            seeds: List of seeds for evaluation
            
        Returns:
            Comparative evaluation results
        """
        self.print_status(f"Comparing {len(agents)} agents")
        
        comparison_results = {}
        
        for agent_name, agent in agents.items():
            self.print_status(f"\n--- Evaluating {agent_name} ---")
            results = self.evaluate_agent(agent, num_episodes, seeds, render=False)
            comparison_results[agent_name] = results
        
        # Generate comparison statistics
        comparison_stats = self._generate_comparison_stats(comparison_results)
        
        return {
            'individual_results': comparison_results,
            'comparison_stats': comparison_stats
        }
    
    def visualize_evaluation(self, 
                           results: Dict[str, Any], 
                           save_path: Optional[Path] = None) -> None:
        """
        Create comprehensive visualizations of evaluation results.
        
        Args:
            results: Results from evaluate_agent or compare_agents
            save_path: Optional path to save the visualization
        """
        if 'individual_results' in results:
            # Multi-agent comparison
            self._visualize_agent_comparison(results, save_path)
        else:
            # Single agent evaluation
            self._visualize_single_agent(results, save_path)
    
    def save_evaluation_report(self, 
                              results: Dict[str, Any], 
                              report_path: Union[str, Path]) -> None:
        """
        Save comprehensive evaluation report to disk.
        
        Args:
            results: Evaluation results
            report_path: Path to save the report
        """
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create comprehensive report
        if 'individual_results' in results:
            # Multi-agent comparison report
            report = self._generate_comparison_report(results)
        else:
            # Single agent report
            report = self._generate_single_agent_report(results)
        
        # Save as JSON
        with open(report_path.with_suffix('.json'), 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Save as text summary
        with open(report_path.with_suffix('.txt'), 'w') as f:
            f.write(self._format_text_report(report))
        
        self.print_status(f"Evaluation report saved to {report_path}")
    
    # Helper methods
    def _calculate_action_entropy(self, action_counts: Dict) -> float:
        """Calculate entropy of action distribution."""
        total_entropy = 0.0
        total_states = 0
        
        for state, actions in action_counts.items():
            total_actions = sum(actions.values())
            if total_actions > 0:
                probs = [count / total_actions for count in actions.values()]
                entropy = -sum(p * np.log2(p) for p in probs if p > 0)
                total_entropy += entropy
                total_states += 1
        
        return total_entropy / max(total_states, 1)
    
    def _aggregate_seed_results(self, seed_results: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate results across different seeds."""
        all_rewards = []
        all_lengths = []
        all_success_rates = []
        all_state_coverages = []
        
        for result in seed_results.values():
            all_rewards.extend(result['rewards'])
            all_lengths.extend(result['lengths'])
            all_success_rates.append(result['statistics']['success_rate'])
            all_state_coverages.append(result['exploration_stats']['state_coverage'])
        
        return {
            'overall_statistics': {
                'mean_reward': np.mean(all_rewards),
                'std_reward': np.std(all_rewards),
                'mean_success_rate': np.mean(all_success_rates),
                'std_success_rate': np.std(all_success_rates),
                'mean_episode_length': np.mean(all_lengths),
                'mean_state_coverage': np.mean(all_state_coverages)
            },
            'seed_consistency': {
                'reward_variance_across_seeds': np.var([
                    result['statistics']['mean_reward'] for result in seed_results.values()
                ]),
                'success_rate_variance': np.var(all_success_rates)
            }
        }
    
    def _generate_comparison_stats(self, comparison_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate statistics comparing multiple agents."""
        agent_stats = {}
        
        for agent_name, results in comparison_results.items():
            stats = results['aggregated_results']['overall_statistics']
            agent_stats[agent_name] = {
                'mean_reward': stats['mean_reward'],
                'success_rate': stats['mean_success_rate'],
                'episode_length': stats['mean_episode_length'],
                'state_coverage': stats['mean_state_coverage']
            }
        
        # Rank agents
        rankings = {
            'by_reward': sorted(agent_stats.items(), key=lambda x: x[1]['mean_reward'], reverse=True),
            'by_success_rate': sorted(agent_stats.items(), key=lambda x: x[1]['success_rate'], reverse=True),
            'by_efficiency': sorted(agent_stats.items(), key=lambda x: x[1]['episode_length']),
            'by_exploration': sorted(agent_stats.items(), key=lambda x: x[1]['state_coverage'], reverse=True)
        }
        
        return {
            'agent_statistics': agent_stats,
            'rankings': rankings
        }
    
    def _visualize_single_agent(self, results: Dict[str, Any], save_path: Optional[Path]) -> None:
        """Create visualizations for single agent evaluation."""
        fig = plt.figure(figsize=(15, 10))
        
        # Extract data
        seed_results = results['per_seed_results']
        all_rewards = []
        all_lengths = []
        
        for result in seed_results.values():
            all_rewards.extend(result['rewards'])
            all_lengths.extend(result['lengths'])
        
        # 1. Reward distribution
        plt.subplot(2, 3, 1)
        plt.hist(all_rewards, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Episode Reward')
        plt.ylabel('Frequency')
        plt.title('Reward Distribution')
        plt.axvline(np.mean(all_rewards), color='red', linestyle='--', label=f'Mean: {np.mean(all_rewards):.1f}')
        plt.legend()
        
        # 2. Episode length distribution
        plt.subplot(2, 3, 2)
        plt.hist(all_lengths, bins=30, alpha=0.7, edgecolor='black', color='orange')
        plt.xlabel('Episode Length')
        plt.ylabel('Frequency')
        plt.title('Episode Length Distribution')
        plt.axvline(np.mean(all_lengths), color='red', linestyle='--', label=f'Mean: {np.mean(all_lengths):.1f}')
        plt.legend()
        
        # 3. Performance across seeds
        plt.subplot(2, 3, 3)
        seed_means = [result['statistics']['mean_reward'] for result in seed_results.values()]
        seed_stds = [result['statistics']['std_reward'] for result in seed_results.values()]
        seeds = [result['seed'] for result in seed_results.values()]
        
        plt.errorbar(seeds, seed_means, yerr=seed_stds, marker='o', capsize=5)
        plt.xlabel('Seed')
        plt.ylabel('Mean Reward')
        plt.title('Performance Across Seeds')
        
        # 4. Success rates
        plt.subplot(2, 3, 4)
        success_rates = [result['statistics']['success_rate'] for result in seed_results.values()]
        timeout_rates = [result['statistics']['timeout_rate'] for result in seed_results.values()]
        failure_rates = [result['statistics']['failure_rate'] for result in seed_results.values()]
        
        x = range(len(seeds))
        plt.bar(x, success_rates, label='Success', alpha=0.7)
        plt.bar(x, timeout_rates, bottom=success_rates, label='Timeout', alpha=0.7)
        plt.bar(x, failure_rates, bottom=np.array(success_rates) + np.array(timeout_rates), 
                label='Failure', alpha=0.7)
        
        plt.xlabel('Seed')
        plt.ylabel('Rate')
        plt.title('Episode Outcomes')
        plt.legend()
        plt.xticks(x, seeds)
        
        # 5. State coverage
        plt.subplot(2, 3, 5)
        coverages = [result['exploration_stats']['state_coverage'] for result in seed_results.values()]
        plt.bar(range(len(seeds)), coverages, alpha=0.7)
        plt.xlabel('Seed')
        plt.ylabel('State Coverage')
        plt.title('Exploration Coverage')
        plt.xticks(range(len(seeds)), seeds)
        
        # 6. Summary statistics
        plt.subplot(2, 3, 6)
        plt.axis('off')
        agg_stats = results['aggregated_results']['overall_statistics']
        summary_text = f"""Summary Statistics:
        
Mean Reward: {agg_stats['mean_reward']:.1f} ± {agg_stats['std_reward']:.1f}
Success Rate: {agg_stats['mean_success_rate']:.1%}
Avg Episode Length: {agg_stats['mean_episode_length']:.1f}
State Coverage: {agg_stats['mean_state_coverage']:.1%}
        """
        plt.text(0.1, 0.8, summary_text, transform=plt.gca().transAxes, 
                fontsize=12, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.print_status(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def _visualize_agent_comparison(self, results: Dict[str, Any], save_path: Optional[Path]) -> None:
        """Create visualizations for multi-agent comparison."""
        fig = plt.figure(figsize=(16, 12))
        
        comparison_stats = results['comparison_stats']
        agent_names = list(comparison_stats['agent_statistics'].keys())
        
        # 1. Mean reward comparison
        plt.subplot(2, 3, 1)
        rewards = [comparison_stats['agent_statistics'][name]['mean_reward'] for name in agent_names]
        bars = plt.bar(agent_names, rewards, alpha=0.7)
        plt.xlabel('Agent')
        plt.ylabel('Mean Reward')
        plt.title('Mean Reward Comparison')
        plt.xticks(rotation=45)
        
        # Add value labels
        for bar, reward in zip(bars, rewards):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f'{reward:.1f}', ha='center', va='bottom')
        
        # 2. Success rate comparison
        plt.subplot(2, 3, 2)
        success_rates = [comparison_stats['agent_statistics'][name]['success_rate'] for name in agent_names]
        bars = plt.bar(agent_names, success_rates, alpha=0.7, color='orange')
        plt.xlabel('Agent')
        plt.ylabel('Success Rate')
        plt.title('Success Rate Comparison')
        plt.xticks(rotation=45)
        
        for bar, rate in zip(bars, success_rates):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{rate:.1%}', ha='center', va='bottom')
        
        # 3. Episode length comparison
        plt.subplot(2, 3, 3)
        lengths = [comparison_stats['agent_statistics'][name]['episode_length'] for name in agent_names]
        bars = plt.bar(agent_names, lengths, alpha=0.7, color='green')
        plt.xlabel('Agent')
        plt.ylabel('Mean Episode Length')
        plt.title('Efficiency Comparison (Lower is Better)')
        plt.xticks(rotation=45)
        
        for bar, length in zip(bars, lengths):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{length:.1f}', ha='center', va='bottom')
        
        # 4. State coverage comparison
        plt.subplot(2, 3, 4)
        coverages = [comparison_stats['agent_statistics'][name]['state_coverage'] for name in agent_names]
        bars = plt.bar(agent_names, coverages, alpha=0.7, color='purple')
        plt.xlabel('Agent')
        plt.ylabel('State Coverage')
        plt.title('Exploration Coverage')
        plt.xticks(rotation=45)
        
        for bar, coverage in zip(bars, coverages):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{coverage:.1%}', ha='center', va='bottom')
        
        # 5. Overall rankings
        plt.subplot(2, 3, 5)
        plt.axis('off')
        ranking_text = "Rankings:\n\n"
        
        for metric, ranking in comparison_stats['rankings'].items():
            ranking_text += f"{metric.replace('_', ' ').title()}:\n"
            for i, (agent, _) in enumerate(ranking):
                ranking_text += f"  {i+1}. {agent}\n"
            ranking_text += "\n"
        
        plt.text(0.05, 0.95, ranking_text, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.print_status(f"Comparison visualization saved to {save_path}")
        
        plt.show()
    
    def _generate_single_agent_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive report for single agent."""
        return {
            'evaluation_summary': results['aggregated_results'],
            'detailed_results': results['per_seed_results'],
            'configuration': results['evaluation_config']
        }
    
    def _generate_comparison_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive report for agent comparison."""
        return {
            'comparison_summary': results['comparison_stats'],
            'individual_agent_results': results['individual_results']
        }
    
    def _format_text_report(self, report: Dict[str, Any]) -> str:
        """Format report as readable text."""
        text = "AGENT EVALUATION REPORT\n"
        text += "=" * 50 + "\n\n"
        
        if 'comparison_summary' in report:
            # Multi-agent comparison
            text += "AGENT COMPARISON\n"
            text += "-" * 20 + "\n\n"
            
            stats = report['comparison_summary']['agent_statistics']
            for agent_name, agent_stats in stats.items():
                text += f"{agent_name}:\n"
                text += f"  Mean Reward: {agent_stats['mean_reward']:.1f}\n"
                text += f"  Success Rate: {agent_stats['success_rate']:.1%}\n"
                text += f"  Episode Length: {agent_stats['episode_length']:.1f}\n"
                text += f"  State Coverage: {agent_stats['state_coverage']:.1%}\n\n"
        
        else:
            # Single agent
            text += "SINGLE AGENT EVALUATION\n"
            text += "-" * 25 + "\n\n"
            
            stats = report['evaluation_summary']['overall_statistics']
            text += f"Mean Reward: {stats['mean_reward']:.1f} ± {stats['std_reward']:.1f}\n"
            text += f"Success Rate: {stats['mean_success_rate']:.1%}\n"
            text += f"Mean Episode Length: {stats['mean_episode_length']:.1f}\n"
            text += f"State Coverage: {stats['mean_state_coverage']:.1%}\n"
        
        return text