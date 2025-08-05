import random
import numpy as np
from collections import defaultdict
from typing import Tuple, Union
from pathlib import Path

from .base import TabularAgent


class DoubleQLearningAgent(TabularAgent):
    """
    Double Q-Learning agent with ε-greedy exploration.
    
    Implements Double Q-Learning to reduce overestimation bias by maintaining
    two separate Q-tables and alternating between them for updates.
    
    Reference: van Hasselt et al. (2010) "Double Q-learning"
    """

    def __init__(
        self,
        obs_space,
        act_space: int,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.2,
        epsilon_decay: float = 0.999,
        epsilon_min: float = 0.05,
    ):
        super().__init__(obs_space, act_space, alpha, gamma, epsilon, epsilon_decay, epsilon_min)
        
        # Two separate Q-tables
        self.Q1 = defaultdict(lambda: np.zeros(self.n_actions, dtype=float))
        self.Q2 = defaultdict(lambda: np.zeros(self.n_actions, dtype=float))
        
        # Combined Q-table for action selection (Q1 + Q2)
        self.Q = defaultdict(lambda: np.zeros(self.n_actions, dtype=float))

    def _update_combined_q(self, state: Tuple[int, int]) -> None:
        """Update the combined Q-table for a given state."""
        self.Q[state] = self.Q1[state] + self.Q2[state]

    def act(self, state: Tuple[int, int], training: bool = True) -> int:
        """
        Choose action using ε-greedy policy based on combined Q-values.
        
        Args:
            state: Current state (row, col)
            training: Whether in training mode (affects exploration)
            
        Returns:
            Action index (0-3)
        """
        # Update combined Q-values for this state
        self._update_combined_q(state)
        
        if training and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        return int(np.argmax(self.Q[state]))

    def learn(
        self, 
        state: Tuple[int, int], 
        action: int, 
        reward: float, 
        next_state: Tuple[int, int], 
        done: bool
    ) -> None:
        """
        Update Q-values using Double Q-Learning update rule.
        
        With probability 0.5:
        - Update Q1 using action selected by Q1 but evaluated by Q2
        Otherwise:
        - Update Q2 using action selected by Q2 but evaluated by Q1
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state reached
            done: Whether episode ended
        """
        # Update combined Q-values for next state
        self._update_combined_q(next_state)
        
        if random.random() < 0.5:
            # Update Q1
            if done:
                target = reward
            else:
                # Select action using Q1, evaluate using Q2
                best_action = np.argmax(self.Q1[next_state])
                target = reward + self.gamma * self.Q2[next_state][best_action]
            
            self.Q1[state][action] += self.alpha * (target - self.Q1[state][action])
        else:
            # Update Q2
            if done:
                target = reward
            else:
                # Select action using Q2, evaluate using Q1
                best_action = np.argmax(self.Q2[next_state])
                target = reward + self.gamma * self.Q1[next_state][best_action]
            
            self.Q2[state][action] += self.alpha * (target - self.Q2[state][action])
        
        # Update combined Q-table for current state
        self._update_combined_q(state)

        if done:
            self.decay_epsilon()

    def save(self, path: Union[str, Path]) -> None:
        """
        Save Double Q-Learning agent to disk.
        
        Args:
            path: File path to save the agent
        """
        agent_data = {
            'q1_table': dict(self.Q1),
            'q2_table': dict(self.Q2),
            'hyperparameters': self.get_hyperparameters(),
            'n_actions': self.n_actions
        }
        
        import pickle
        with open(path, "wb") as f:
            pickle.dump(agent_data, f)

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'DoubleQLearningAgent':
        """
        Load Double Q-Learning agent from disk.
        
        Args:
            path: File path to load the agent from
            
        Returns:
            Loaded DoubleQLearningAgent instance
        """
        import pickle
        with open(path, "rb") as f:
            agent_data = pickle.load(f)
        
        if not isinstance(agent_data, dict) or 'q1_table' not in agent_data:
            raise ValueError("Invalid save file format for Double Q-Learning agent")
        
        q1_table = agent_data['q1_table']
        q2_table = agent_data['q2_table']
        n_actions = agent_data['n_actions']
        hyperparams = agent_data.get('hyperparameters', {})
        
        # Create agent instance
        agent = cls(obs_space=None, act_space=n_actions, **hyperparams)
        
        # Restore Q-tables
        agent.Q1 = defaultdict(lambda: np.zeros(n_actions, dtype=float))
        agent.Q1.update(q1_table)
        agent.Q2 = defaultdict(lambda: np.zeros(n_actions, dtype=float))
        agent.Q2.update(q2_table)
        
        # Rebuild combined Q-table
        all_states = set(q1_table.keys()) | set(q2_table.keys())
        for state in all_states:
            agent._update_combined_q(state)
        
        return agent

    def get_q_statistics(self) -> dict:
        """
        Get statistics about the Q-tables for analysis.
        
        Returns:
            Dictionary with Q-table statistics
        """
        all_states = set(self.Q1.keys()) | set(self.Q2.keys())
        
        if not all_states:
            return {'num_states': 0}
        
        q1_values = []
        q2_values = []
        combined_values = []
        
        for state in all_states:
            q1_values.extend(self.Q1[state])
            q2_values.extend(self.Q2[state])
            combined_values.extend(self.Q[state])
        
        return {
            'num_states': len(all_states),
            'q1_stats': {
                'mean': np.mean(q1_values),
                'std': np.std(q1_values),
                'min': np.min(q1_values),
                'max': np.max(q1_values)
            },
            'q2_stats': {
                'mean': np.mean(q2_values),
                'std': np.std(q2_values),
                'min': np.min(q2_values),
                'max': np.max(q2_values)
            },
            'combined_stats': {
                'mean': np.mean(combined_values),
                'std': np.std(combined_values),
                'min': np.min(combined_values),
                'max': np.max(combined_values)
            }
        }