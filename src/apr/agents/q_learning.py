import random
import numpy as np
from collections import defaultdict
from typing import Tuple, Union
from pathlib import Path

from .base import TabularAgent


class QLearningAgent(TabularAgent):
    """
    Q-Learning agent with ε-greedy exploration for discrete states & actions.
    
    Implements the classic off-policy temporal difference learning algorithm.
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
        self.Q = defaultdict(lambda: np.zeros(self.n_actions, dtype=float))

    # ------------------------------------------------------------------ #

    def act(self, state: Tuple[int, int], training: bool = True) -> int:
        """
        Choose action using ε-greedy policy.
        
        Args:
            state: Current state (row, col)
            training: Whether in training mode (affects exploration)
            
        Returns:
            Action index (0-3)
        """
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
        Update Q-values using Q-Learning update rule.
        
        Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state reached
            done: Whether episode ended
        """
        best_next = np.max(self.Q[next_state])
        target = reward + (0 if done else self.gamma * best_next)
        self.Q[state][action] += self.alpha * (target - self.Q[state][action])

        if done:
            self.decay_epsilon()

    def save(self, path: Union[str, Path]) -> None:
        """
        Save Q-Learning agent to disk.
        
        Args:
            path: File path to save the agent
        """
        agent_data = {
            'q_table': dict(self.Q),
            'hyperparameters': self.get_hyperparameters(),
            'n_actions': self.n_actions
        }
        
        import pickle
        with open(path, "wb") as f:
            pickle.dump(agent_data, f)

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'QLearningAgent':
        """
        Load Q-Learning agent from disk.
        
        Args:
            path: File path to load the agent from
            
        Returns:
            Loaded QLearningAgent instance
        """
        import pickle
        with open(path, "rb") as f:
            agent_data = pickle.load(f)
        
        # Handle both old and new save formats
        if isinstance(agent_data, dict) and 'q_table' in agent_data:
            # New format
            q_table = agent_data['q_table']
            n_actions = agent_data['n_actions']
            hyperparams = agent_data.get('hyperparameters', {})
        else:
            # Old format (backward compatibility)
            q_table = agent_data
            n_actions = len(next(iter(q_table.values())))
            hyperparams = {}
        
        # Create agent instance
        agent = cls(obs_space=None, act_space=n_actions, **hyperparams)
        agent.Q = defaultdict(lambda: np.zeros(n_actions, dtype=float))
        agent.Q.update(q_table)
        
        return agent