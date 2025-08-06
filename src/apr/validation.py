"""
Comprehensive validation framework for reinforcement learning agents.

This module provides rigorous testing to ensure agents are truly learning,
not memorizing, and can generalize across different scenarios.
"""

import copy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict, Counter
import scipy.stats as stats
from pathlib import Path

from .env import WarehouseEnv
from .agents.base import BaseAgent
from .train import run_episode


class RLAgentValidator:
    """
    Comprehensive validation suite for RL agents.
    
    Tests for:
    - Actual learning vs random behavior
    - Convergence and stability
    - Exploration vs exploitation balance
    - Generalization across scenarios
    - Local optima avoidance
    - Statistical significance of results
    """
    
    def __init__(self, agent: BaseAgent, env: WarehouseEnv, verbose: bool = True):
        self.agent = agent
        self.env = env
        self.verbose = verbose
        self.validation_results = {}
        
    def print_status(self, message: str):
        """Print status if verbose mode is enabled."""
        if self.verbose:
            print(message)
    
    def full_validation(self, 
                       training_episodes: int = 500,
                       test_episodes: int = 100,
                       n_seeds: int = 5) -> Dict[str, Any]:
        """
        Run complete validation suite.
        
        Args:
            training_episodes: Episodes for training
            test_episodes: Episodes for testing
            n_seeds: Number of different seeds to test
            
        Returns:
            Comprehensive validation results
        """
        self.print_status("🔍 Starting Comprehensive RL Agent Validation")
        self.print_status("=" * 60)
        
        results = {}
        
        # 1. Learning Verification
        self.print_status("\n1️⃣ Testing if agent is actually learning...")
        results['learning'] = self.test_learning_vs_random(training_episodes, test_episodes)
        
        # 2. Convergence Analysis
        self.print_status("\n2️⃣ Analyzing convergence and stability...")
        results['convergence'] = self.analyze_convergence(training_episodes)
        
        # 3. Exploration Analysis
        self.print_status("\n3️⃣ Checking exploration vs exploitation...")
        results['exploration'] = self.analyze_exploration(training_episodes)
        
        # 4. Generalization Testing
        self.print_status("\n4️⃣ Testing generalization across scenarios...")
        results['generalization'] = self.test_generalization(n_seeds, test_episodes)
        
        # 5. Local Optima Detection
        self.print_status("\n5️⃣ Checking for local optima...")
        results['local_optima'] = self.detect_local_optima(training_episodes)
        
        # 6. Policy Analysis
        self.print_status("\n6️⃣ Analyzing learned policy...")
        results['policy'] = self.analyze_policy()
        
        # 7. Generate comprehensive report
        self.print_status("\n7️⃣ Generating validation report...")
        results['summary'] = self.generate_summary_report(results)
        
        self.validation_results = results
        self.print_status("\n✅ Validation complete!")
        return results
    
    def test_learning_vs_random(self, train_episodes: int, test_episodes: int) -> Dict[str, Any]:
        """Test if agent performs significantly better than random policy."""
        
        # Train the agent
        self.print_status("   Training agent...")
        train_rewards = []
        for episode in range(train_episodes):
            reward = run_episode(self.env, self.agent, training=True)
            train_rewards.append(reward)
            
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(train_rewards[-100:])
                self.print_status(f"   Episode {episode + 1}: Avg Reward = {avg_reward:.1f}")
        
        # Test trained agent
        self.print_status("   Testing trained agent...")
        trained_rewards = []
        for _ in range(test_episodes):
            reward = run_episode(self.env, self.agent, training=False)
            trained_rewards.append(reward)
        
        # Test random agent
        self.print_status("   Testing random baseline...")
        random_rewards = []
        for _ in range(test_episodes):
            self.env.reset()
            total_reward = 0
            done = False
            steps = 0
            
            while not done and steps < self.env.max_steps:
                action = np.random.randint(0, self.env.n_actions)
                _, reward, done, _ = self.env.step(action)
                total_reward += reward
                steps += 1
                
            random_rewards.append(total_reward)
        
        # Statistical analysis
        trained_mean = np.mean(trained_rewards)
        random_mean = np.mean(random_rewards)
        
        # T-test for significance
        t_stat, p_value = stats.ttest_ind(trained_rewards, random_rewards)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((np.std(trained_rewards)**2 + np.std(random_rewards)**2) / 2))
        cohens_d = (trained_mean - random_mean) / pooled_std
        
        learning_result = {
            'trained_performance': {
                'mean': trained_mean,
                'std': np.std(trained_rewards),
                'rewards': trained_rewards
            },
            'random_performance': {
                'mean': random_mean,
                'std': np.std(random_rewards),
                'rewards': random_rewards
            },
            'statistical_test': {
                't_statistic': t_stat,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'significant': p_value < 0.05
            },
            'improvement': trained_mean - random_mean,
            'improvement_percent': ((trained_mean - random_mean) / abs(random_mean)) * 100 if random_mean != 0 else float('inf')
        }
        
        self.print_status(f"   Trained: {trained_mean:.1f} ± {np.std(trained_rewards):.1f}")
        self.print_status(f"   Random:  {random_mean:.1f} ± {np.std(random_rewards):.1f}")
        self.print_status(f"   Improvement: {learning_result['improvement']:.1f} ({learning_result['improvement_percent']:.1f}%)")
        self.print_status(f"   Statistically significant: {learning_result['statistical_test']['significant']}")
        
        return learning_result
    
    def analyze_convergence(self, episodes: int) -> Dict[str, Any]:
        """Analyze learning convergence and stability."""
        
        # Fresh agent for convergence analysis
        convergence_agent = copy.deepcopy(self.agent)
        
        rewards = []
        q_value_changes = []
        policy_changes = []
        
        prev_q_table = None
        prev_policy = None
        
        window_size = 50
        
        for episode in range(episodes):
            # Store previous state for comparison
            if hasattr(convergence_agent, 'Q'):
                if prev_q_table is not None:
                    # Calculate Q-value changes
                    q_change = self._calculate_q_table_change(prev_q_table, convergence_agent.Q)
                    q_value_changes.append(q_change)
                
                prev_q_table = copy.deepcopy(dict(convergence_agent.Q))
                
                # Calculate policy changes
                current_policy = self._extract_policy(convergence_agent)
                if prev_policy is not None:
                    policy_change = self._calculate_policy_change(prev_policy, current_policy)
                    policy_changes.append(policy_change)
                prev_policy = current_policy
            
            # Run episode
            reward = run_episode(self.env, convergence_agent, training=True)
            rewards.append(reward)
            
            # Check for convergence every window
            if (episode + 1) % window_size == 0 and len(rewards) >= window_size * 2:
                recent_avg = np.mean(rewards[-window_size:])
                prev_avg = np.mean(rewards[-2*window_size:-window_size])
                self.print_status(f"   Episode {episode + 1}: Recent={recent_avg:.1f}, Previous={prev_avg:.1f}")
        
        # Analyze convergence metrics
        convergence_result = {
            'rewards': rewards,
            'q_value_changes': q_value_changes,
            'policy_changes': policy_changes,
            'convergence_analysis': self._analyze_convergence_patterns(rewards, q_value_changes, policy_changes),
            'stability_metrics': self._calculate_stability_metrics(rewards)
        }
        
        return convergence_result
    
    def analyze_exploration(self, episodes: int) -> Dict[str, Any]:
        """Analyze exploration vs exploitation behavior."""
        
        exploration_agent = copy.deepcopy(self.agent)
        
        state_visits = defaultdict(int)
        action_counts = defaultdict(lambda: defaultdict(int))
        epsilon_values = []
        state_action_pairs = set()
        
        for episode in range(episodes):
            self.env.reset()
            done = False
            steps = 0
            
            while not done and steps < self.env.max_steps:
                state = self.env.agent_pos
                action = exploration_agent.act(state, training=True)
                
                # Track exploration metrics
                state_visits[state] += 1
                action_counts[state][action] += 1
                state_action_pairs.add((state, action))
                
                if hasattr(exploration_agent, 'epsilon'):
                    epsilon_values.append(exploration_agent.epsilon)
                
                # Take step
                next_state, reward, done, _ = self.env.step(action)
                exploration_agent.learn(state, action, reward, next_state, done)
                steps += 1
        
        # Calculate exploration metrics
        total_possible_states = self.env.n_rows * self.env.n_cols
        total_possible_sa_pairs = total_possible_states * self.env.n_actions
        
        exploration_result = {
            'state_coverage': {
                'visited_states': len(state_visits),
                'total_possible': total_possible_states,
                'coverage_percent': (len(state_visits) / total_possible_states) * 100,
                'state_visits': dict(state_visits)
            },
            'action_diversity': {
                'unique_sa_pairs': len(state_action_pairs),
                'total_possible': total_possible_sa_pairs,
                'diversity_percent': (len(state_action_pairs) / total_possible_sa_pairs) * 100,
                'action_entropy': self._calculate_action_entropy(action_counts)
            },
            'epsilon_decay': {
                'values': epsilon_values,
                'final_epsilon': epsilon_values[-1] if epsilon_values else None,
                'decay_rate': self._calculate_epsilon_decay_rate(epsilon_values)
            }
        }
        
        self.print_status(f"   State coverage: {exploration_result['state_coverage']['coverage_percent']:.1f}%")
        self.print_status(f"   Action diversity: {exploration_result['action_diversity']['diversity_percent']:.1f}%")
        
        return exploration_result
    
    def test_generalization(self, n_seeds: int, test_episodes: int) -> Dict[str, Any]:
        """Test agent performance across different random seeds and scenarios."""
        
        performances = []
        
        for seed in range(n_seeds):
            self.print_status(f"   Testing seed {seed + 1}/{n_seeds}...")
            
            # Create environment with different seed
            test_env = WarehouseEnv(seed=seed)
            
            # Test agent performance
            seed_rewards = []
            for _ in range(test_episodes):
                reward = run_episode(test_env, self.agent, training=False)
                seed_rewards.append(reward)
            
            performances.append({
                'seed': seed,
                'rewards': seed_rewards,
                'mean': np.mean(seed_rewards),
                'std': np.std(seed_rewards)
            })
        
        # Analyze generalization
        all_means = [p['mean'] for p in performances]
        all_stds = [p['std'] for p in performances]
        
        generalization_result = {
            'performances': performances,
            'overall_mean': np.mean(all_means),
            'mean_variance': np.var(all_means),
            'consistency_score': 1.0 / (1.0 + np.var(all_means)),  # Higher is better
            'statistical_analysis': {
                'anova_f': None,
                'anova_p': None
            }
        }
        
        # ANOVA test for significant differences between seeds
        if n_seeds > 2:
            all_rewards = [p['rewards'] for p in performances]
            f_stat, p_val = stats.f_oneway(*all_rewards)
            generalization_result['statistical_analysis']['anova_f'] = f_stat
            generalization_result['statistical_analysis']['anova_p'] = p_val
        
        self.print_status(f"   Mean performance: {generalization_result['overall_mean']:.1f}")
        self.print_status(f"   Consistency score: {generalization_result['consistency_score']:.3f}")
        
        return generalization_result
    
    def detect_local_optima(self, episodes: int) -> Dict[str, Any]:
        """Detect if agent is stuck in local optima."""
        
        # Train multiple agents with different initializations
        n_runs = 3
        final_performances = []
        learning_curves = []
        
        for run in range(n_runs):
            self.print_status(f"   Training run {run + 1}/{n_runs}...")
            
            # Fresh agent with different random initialization
            test_agent = copy.deepcopy(self.agent)
            
            # Reset epsilon for proper exploration
            if hasattr(test_agent, 'epsilon'):
                test_agent.epsilon = self.agent.epsilon
            
            rewards = []
            for episode in range(episodes):
                reward = run_episode(self.env, test_agent, training=True)
                rewards.append(reward)
            
            learning_curves.append(rewards)
            final_performances.append(np.mean(rewards[-50:]))  # Last 50 episodes
        
        # Analyze for local optima
        performance_variance = np.var(final_performances)
        performance_range = max(final_performances) - min(final_performances)
        
        local_optima_result = {
            'final_performances': final_performances,
            'learning_curves': learning_curves,
            'performance_variance': performance_variance,
            'performance_range': performance_range,
            'likely_local_optima': performance_range > 50,  # Threshold for concern
            'consistency_across_runs': performance_variance < 100  # Lower variance is better
        }
        
        self.print_status(f"   Performance range: {performance_range:.1f}")
        self.print_status(f"   Likely local optima: {local_optima_result['likely_local_optima']}")
        
        return local_optima_result
    
    def analyze_policy(self) -> Dict[str, Any]:
        """Analyze the learned policy for reasonableness."""
        
        if not hasattr(self.agent, 'Q'):
            return {'error': 'Agent does not have Q-table for policy analysis'}
        
        # Extract policy from Q-table
        policy = {}
        q_values_analysis = {}
        
        for state, q_vals in self.agent.Q.items():
            if isinstance(q_vals, np.ndarray):
                best_action = np.argmax(q_vals)
                policy[state] = best_action
                q_values_analysis[state] = {
                    'q_values': q_vals.tolist(),
                    'best_action': int(best_action),
                    'q_variance': np.var(q_vals),
                    'max_q': np.max(q_vals),
                    'min_q': np.min(q_vals)
                }
        
        # Analyze policy characteristics
        action_distribution = Counter(policy.values())
        
        policy_result = {
            'policy': policy,
            'q_values_analysis': q_values_analysis,
            'action_distribution': dict(action_distribution),
            'policy_entropy': self._calculate_policy_entropy(policy),
            'value_function_stats': self._analyze_value_function(q_values_analysis)
        }
        
        return policy_result
    
    def generate_summary_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive summary report."""
        
        summary = {
            'overall_assessment': 'PASS',
            'critical_issues': [],
            'warnings': [],
            'recommendations': [],
            'scores': {}
        }
        
        # Learning validation
        if results['learning']['statistical_test']['significant']:
            summary['scores']['learning'] = 'PASS'
        else:
            summary['scores']['learning'] = 'FAIL'
            summary['critical_issues'].append("Agent does not perform significantly better than random")
            summary['overall_assessment'] = 'FAIL'
        
        # Convergence validation
        convergence_score = results['convergence']['stability_metrics']['coefficient_of_variation']
        if convergence_score < 0.5:  # Low variance in recent performance
            summary['scores']['convergence'] = 'PASS'
        else:
            summary['scores']['convergence'] = 'WARNING'
            summary['warnings'].append("High variance in performance - may not have converged")
        
        # Exploration validation
        state_coverage = results['exploration']['state_coverage']['coverage_percent']
        if state_coverage > 50:
            summary['scores']['exploration'] = 'PASS'
        else:
            summary['scores']['exploration'] = 'WARNING'
            summary['warnings'].append(f"Low state coverage ({state_coverage:.1f}%) - may be under-exploring")
        
        # Generalization validation
        consistency = results['generalization']['consistency_score']
        if consistency > 0.5:
            summary['scores']['generalization'] = 'PASS'
        else:
            summary['scores']['generalization'] = 'WARNING'
            summary['warnings'].append("Inconsistent performance across different scenarios")
        
        # Local optima validation
        if not results['local_optima']['likely_local_optima']:
            summary['scores']['local_optima'] = 'PASS'
        else:
            summary['scores']['local_optima'] = 'WARNING'
            summary['warnings'].append("Agent may be stuck in local optima")
        
        # Overall assessment
        if summary['critical_issues']:
            summary['overall_assessment'] = 'FAIL'
        elif summary['warnings']:
            summary['overall_assessment'] = 'PASS_WITH_WARNINGS'
        else:
            summary['overall_assessment'] = 'PASS'
        
        return summary
    
    def visualize_results(self, save_path: Optional[Path] = None):
        """Create comprehensive visualization of validation results."""
        
        if not self.validation_results:
            print("No validation results to visualize. Run full_validation() first.")
            return
        
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Learning curves comparison
        plt.subplot(3, 4, 1)
        if 'learning' in self.validation_results:
            learning_data = self.validation_results['learning']
            trained_rewards = learning_data['trained_performance']['rewards']
            random_rewards = learning_data['random_performance']['rewards']
            
            plt.hist(trained_rewards, alpha=0.7, label='Trained Agent', bins=20)
            plt.hist(random_rewards, alpha=0.7, label='Random Agent', bins=20)
            plt.xlabel('Reward')
            plt.ylabel('Frequency')
            plt.title('Performance Comparison')
            plt.legend()
        
        # 2. Convergence analysis
        plt.subplot(3, 4, 2)
        if 'convergence' in self.validation_results:
            rewards = self.validation_results['convergence']['rewards']
            window = 50
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            plt.plot(smoothed)
            plt.xlabel('Episode')
            plt.ylabel('Reward (smoothed)')
            plt.title('Learning Convergence')
        
        # 3. State coverage heatmap
        plt.subplot(3, 4, 3)
        if 'exploration' in self.validation_results:
            state_visits = self.validation_results['exploration']['state_coverage']['state_visits']
            coverage_grid = np.zeros((self.env.n_rows, self.env.n_cols))
            for (r, c), visits in state_visits.items():
                if 0 <= r < self.env.n_rows and 0 <= c < self.env.n_cols:
                    coverage_grid[r, c] = visits
            
            sns.heatmap(coverage_grid, annot=True, fmt='.0f', cmap='Blues')
            plt.title('State Visitation Heatmap')
        
        # 4. Generalization across seeds
        plt.subplot(3, 4, 4)
        if 'generalization' in self.validation_results:
            performances = self.validation_results['generalization']['performances']
            means = [p['mean'] for p in performances]
            stds = [p['std'] for p in performances]
            seeds = [p['seed'] for p in performances]
            
            plt.errorbar(seeds, means, yerr=stds, marker='o', capsize=5)
            plt.xlabel('Seed')
            plt.ylabel('Mean Reward')
            plt.title('Generalization Across Seeds')
        
        # 5. Q-value changes over time
        plt.subplot(3, 4, 5)
        if 'convergence' in self.validation_results and self.validation_results['convergence']['q_value_changes']:
            q_changes = self.validation_results['convergence']['q_value_changes']
            plt.plot(q_changes)
            plt.xlabel('Episode')
            plt.ylabel('Q-value Change')
            plt.title('Q-value Convergence')
            plt.yscale('log')
        
        # 6. Policy stability
        plt.subplot(3, 4, 6)
        if 'convergence' in self.validation_results and self.validation_results['convergence']['policy_changes']:
            policy_changes = self.validation_results['convergence']['policy_changes']
            plt.plot(policy_changes)
            plt.xlabel('Episode')
            plt.ylabel('Policy Change %')
            plt.title('Policy Stability')
        
        # 7. Action distribution
        plt.subplot(3, 4, 7)
        if 'policy' in self.validation_results and 'action_distribution' in self.validation_results['policy']:
            action_dist = self.validation_results['policy']['action_distribution']
            actions = list(action_dist.keys())
            counts = list(action_dist.values())
            action_names = ['Up', 'Down', 'Left', 'Right']
            
            plt.bar([action_names[a] for a in actions], counts)
            plt.xlabel('Action')
            plt.ylabel('Frequency in Policy')
            plt.title('Learned Policy Distribution')
        
        # 8. Local optima analysis
        plt.subplot(3, 4, 8)
        if 'local_optima' in self.validation_results:
            learning_curves = self.validation_results['local_optima']['learning_curves']
            for i, curve in enumerate(learning_curves):
                window = 50
                smoothed = np.convolve(curve, np.ones(window)/window, mode='valid')
                plt.plot(smoothed, label=f'Run {i+1}', alpha=0.7)
            plt.xlabel('Episode')
            plt.ylabel('Reward (smoothed)')
            plt.title('Multiple Training Runs')
            plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
    
    # Helper methods
    def _calculate_q_table_change(self, prev_q: Dict, curr_q: Dict) -> float:
        """Calculate the magnitude of change in Q-table."""
        total_change = 0.0
        count = 0
        
        for state in prev_q:
            if state in curr_q:
                if isinstance(prev_q[state], np.ndarray) and isinstance(curr_q[state], np.ndarray):
                    change = np.sum(np.abs(curr_q[state] - prev_q[state]))
                    total_change += change
                    count += 1
        
        return total_change / max(count, 1)
    
    def _extract_policy(self, agent) -> Dict:
        """Extract greedy policy from agent."""
        if not hasattr(agent, 'Q'):
            return {}
        
        policy = {}
        for state, q_vals in agent.Q.items():
            if isinstance(q_vals, np.ndarray):
                policy[state] = np.argmax(q_vals)
        return policy
    
    def _calculate_policy_change(self, prev_policy: Dict, curr_policy: Dict) -> float:
        """Calculate percentage of policy that changed."""
        if not prev_policy or not curr_policy:
            return 0.0
        
        common_states = set(prev_policy.keys()) & set(curr_policy.keys())
        if not common_states:
            return 0.0
        
        changes = sum(1 for state in common_states if prev_policy[state] != curr_policy[state])
        return (changes / len(common_states)) * 100
    
    def _analyze_convergence_patterns(self, rewards: List[float], 
                                     q_changes: List[float], 
                                     policy_changes: List[float]) -> Dict[str, Any]:
        """Analyze convergence patterns in the data."""
        analysis = {}
        
        if rewards:
            # Reward trend analysis
            window = min(100, len(rewards) // 4)
            if len(rewards) >= window * 2:
                early_rewards = rewards[:window]
                late_rewards = rewards[-window:]
                analysis['reward_improvement'] = np.mean(late_rewards) - np.mean(early_rewards)
                analysis['reward_trend'] = 'improving' if analysis['reward_improvement'] > 0 else 'declining'
        
        if q_changes:
            # Q-value convergence
            analysis['q_converged'] = len(q_changes) > 50 and np.mean(q_changes[-50:]) < np.mean(q_changes[:50])
            analysis['final_q_change_rate'] = np.mean(q_changes[-20:]) if len(q_changes) >= 20 else None
        
        if policy_changes:
            # Policy stability
            analysis['policy_stable'] = len(policy_changes) > 50 and np.mean(policy_changes[-50:]) < 5.0
            analysis['final_policy_change_rate'] = np.mean(policy_changes[-20:]) if len(policy_changes) >= 20 else None
        
        return analysis
    
    def _calculate_stability_metrics(self, rewards: List[float]) -> Dict[str, float]:
        """Calculate stability metrics for reward sequence."""
        if len(rewards) < 100:
            return {'error': 'Insufficient data for stability analysis'}
        
        recent_rewards = rewards[-100:]  # Last 100 episodes
        
        return {
            'mean': np.mean(recent_rewards),
            'std': np.std(recent_rewards),
            'coefficient_of_variation': np.std(recent_rewards) / np.mean(recent_rewards) if np.mean(recent_rewards) != 0 else float('inf'),
            'min': np.min(recent_rewards),
            'max': np.max(recent_rewards),
            'range': np.max(recent_rewards) - np.min(recent_rewards)
        }
    
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
    
    def _calculate_epsilon_decay_rate(self, epsilon_values: List[float]) -> float:
        """Calculate the effective epsilon decay rate."""
        if len(epsilon_values) < 2:
            return 0.0
        
        # Calculate average decay rate
        decay_rates = []
        for i in range(1, len(epsilon_values)):
            if epsilon_values[i-1] > 0:
                decay_rate = epsilon_values[i] / epsilon_values[i-1]
                decay_rates.append(decay_rate)
        
        return np.mean(decay_rates) if decay_rates else 1.0
    
    def _calculate_policy_entropy(self, policy: Dict) -> float:
        """Calculate entropy of the policy distribution."""
        if not policy:
            return 0.0
        
        action_counts = Counter(policy.values())
        total_states = len(policy)
        
        probs = [count / total_states for count in action_counts.values()]
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        
        return entropy
    
    def _analyze_value_function(self, q_analysis: Dict) -> Dict[str, Any]:
        """Analyze the learned value function."""
        if not q_analysis:
            return {}
        
        all_q_values = []
        all_variances = []
        all_ranges = []
        
        for state_data in q_analysis.values():
            q_vals = state_data['q_values']
            all_q_values.extend(q_vals)
            all_variances.append(state_data['q_variance'])
            all_ranges.append(state_data['max_q'] - state_data['min_q'])
        
        return {
            'overall_q_mean': np.mean(all_q_values),
            'overall_q_std': np.std(all_q_values),
            'overall_q_range': max(all_q_values) - min(all_q_values),
            'average_state_variance': np.mean(all_variances),
            'average_state_range': np.mean(all_ranges),
            'value_function_spread': np.std(all_q_values)
        }