"""
Base agent interface for all reinforcement learning algorithms.

This module defines the abstract base class that all RL agents must implement,
ensuring consistent interfaces and making it easy to swap algorithms.
"""

from abc import ABC, abstractmethod
from typing import Any, Tuple, Union
import pickle
from pathlib import Path


class BaseAgent(ABC):
    """
    Abstract base class for all RL agents in the APR framework.
    
    All concrete agents (Q-Learning, SARSA, Double Q-Learning, etc.) must
    inherit from this class and implement the required methods.
    """

    def __init__(
        self,
        obs_space: Any,
        act_space: int,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.3,
        epsilon_decay: float = 0.999,
        epsilon_min: float = 0.05
    ):
        """
        Initialize base agent with common hyperparameters.
        
        Args:
            obs_space: Observation space (environment dependent)
            act_space: Number of actions available
            alpha: Learning rate
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_decay: Epsilon decay rate per episode
            epsilon_min: Minimum epsilon value
        """
        self.obs_space = obs_space
        self.n_actions = act_space
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    @abstractmethod
    def act(self, state: Tuple[int, int], training: bool = True) -> int:
        """
        Choose an action given the current state.
        
        Args:
            state: Current state (row, col)
            training: Whether in training mode (affects exploration)
            
        Returns:
            Action index (0-3 for up, down, left, right)
        """
        pass

    @abstractmethod
    def learn(
        self, 
        state: Tuple[int, int], 
        action: int, 
        reward: float, 
        next_state: Tuple[int, int], 
        done: bool
    ) -> None:
        """
        Update the agent's policy based on experience.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state reached
            done: Whether episode ended
        """
        pass

    @abstractmethod
    def save(self, path: Union[str, Path]) -> None:
        """
        Save the agent's learned parameters to disk.
        
        Args:
            path: File path to save the agent
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: Union[str, Path]) -> 'BaseAgent':
        """
        Load a previously saved agent from disk.
        
        Args:
            path: File path to load the agent from
            
        Returns:
            Loaded agent instance
        """
        pass

    def decay_epsilon(self) -> None:
        """
        Decay epsilon after each episode (common to most algorithms).
        """
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def get_hyperparameters(self) -> dict:
        """
        Get current hyperparameters as a dictionary.
        
        Returns:
            Dictionary of hyperparameter names and values
        """
        return {
            'alpha': self.alpha,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'epsilon_decay': self.epsilon_decay,
            'epsilon_min': self.epsilon_min
        }

    def set_hyperparameters(self, **kwargs) -> None:
        """
        Update hyperparameters from keyword arguments.
        
        Args:
            **kwargs: Hyperparameter name-value pairs
        """
        for param, value in kwargs.items():
            if hasattr(self, param):
                setattr(self, param, value)
            else:
                raise ValueError(f"Unknown hyperparameter: {param}")

    def __repr__(self) -> str:
        """String representation of the agent."""
        return f"{self.__class__.__name__}(alpha={self.alpha}, gamma={self.gamma}, epsilon={self.epsilon:.3f})"


class TabularAgent(BaseAgent):
    """
    Base class for tabular RL agents (Q-Learning, SARSA, etc.).
    
    Provides common functionality for agents that use lookup tables
    to store state-action values.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Will be initialized by concrete classes
        self.Q = None

    def get_state_values(self, state: Tuple[int, int]) -> dict:
        """
        Get Q-values for all actions in a given state.
        
        Args:
            state: State to query
            
        Returns:
            Dictionary mapping actions to Q-values
        """
        if self.Q is None:
            raise RuntimeError("Q-table not initialized")
        
        return dict(enumerate(self.Q[state]))

    def get_best_action(self, state: Tuple[int, int]) -> int:
        """
        Get the best action for a state (greedy policy).
        
        Args:
            state: State to query
            
        Returns:
            Best action index
        """
        if self.Q is None:
            raise RuntimeError("Q-table not initialized")
            
        return int(self.Q[state].argmax())

    def get_state_value(self, state: Tuple[int, int]) -> float:
        """
        Get the value of a state (max Q-value).
        
        Args:
            state: State to query
            
        Returns:
            State value
        """
        if self.Q is None:
            raise RuntimeError("Q-table not initialized")
            
        return float(self.Q[state].max())

    def save_q_table(self, path: Union[str, Path]) -> None:
        """
        Save Q-table to a pickle file.
        
        Args:
            path: File path to save Q-table
        """
        if self.Q is None:
            raise RuntimeError("Q-table not initialized")
            
        with open(path, 'wb') as f:
            pickle.dump(dict(self.Q), f)

    def load_q_table(self, path: Union[str, Path]) -> None:
        """
        Load Q-table from a pickle file.
        
        Args:
            path: File path to load Q-table from
        """
        with open(path, 'rb') as f:
            q_data = pickle.load(f)
            
        # Convert back to defaultdict format
        from collections import defaultdict
        import numpy as np
        
        self.Q = defaultdict(lambda: np.zeros(self.n_actions, dtype=float))
        self.Q.update(q_data)