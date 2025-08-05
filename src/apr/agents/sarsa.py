"""
SARSA agent implementation for autonomous parcel routing.

SARSA (State-Action-Reward-State-Action) is an on-policy temporal difference 
learning algorithm that learns the Q-values for the policy it is following.
"""

import random
import numpy as np
from collections import defaultdict
from typing import Tuple, Union
from pathlib import Path

from .base import TabularAgent


class SarsaAgent(TabularAgent):
    """
    SARSA agent with ε-greedy exploration for discrete states & actions.
    
    Implements the on-policy temporal difference learning algorithm.
    Unlike Q-Learning, SARSA updates based on the actual next action taken,
    making it more conservative in stochastic environments.
    """

    def __init__(
        self,
        obs_space,
        act_space: int,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.3,
        epsilon_decay: float = 0.95,
        epsilon_min: float = 0.01,
    ):
        super().__init__(obs_space, act_space, alpha, gamma, epsilon, epsilon_decay, epsilon_min)
        self.Q = defaultdict(lambda: np.zeros(self.n_actions, dtype=float))
        
        # SARSA needs to remember the next action for learning
        self.next_action = None

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
        else:
            # Greedy action selection
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
        Update Q-values using SARSA update rule.
        
        SARSA: Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]
        where a' is the actual next action that will be taken.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state reached
            done: Whether episode ended
        """
        if done:
            # Terminal state - no next action
            target = reward
        else:
            # Choose next action using current policy
            next_action = self.act(next_state, training=True)
            target = reward + self.gamma * self.Q[next_state][next_action]
        
        # SARSA update
        self.Q[state][action] += self.alpha * (target - self.Q[state][action])

        if done:
            self.decay_epsilon()

    def save(self, path: Union[str, Path]) -> None:
        """
        Save SARSA agent to disk.
        
        Args:
            path: File path to save the agent
        """
        agent_data = {
            'q_table': dict(self.Q),
            'hyperparameters': self.get_hyperparameters(),
            'n_actions': self.n_actions,
            'algorithm': 'sarsa'
        }
        
        import pickle
        with open(path, "wb") as f:
            pickle.dump(agent_data, f)

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'SarsaAgent':
        """
        Load SARSA agent from disk.
        
        Args:
            path: File path to load the agent from
            
        Returns:
            Loaded SarsaAgent instance
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


class SarsaLambdaAgent(SarsaAgent):
    """
    SARSA(λ) agent with eligibility traces for faster learning.
    
    Extends SARSA with eligibility traces that allow updates to states
    visited earlier in the episode, leading to faster convergence.
    """

    def __init__(
        self,
        obs_space,
        act_space: int,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.3,
        epsilon_decay: float = 0.95,
        epsilon_min: float = 0.01,
        lambda_trace: float = 0.9,
    ):
        super().__init__(obs_space, act_space, alpha, gamma, epsilon, epsilon_decay, epsilon_min)
        self.lambda_trace = lambda_trace
        self.eligibility_traces = defaultdict(lambda: np.zeros(self.n_actions, dtype=float))

    def learn(
        self, 
        state: Tuple[int, int], 
        action: int, 
        reward: float, 
        next_state: Tuple[int, int], 
        done: bool
    ) -> None:
        """
        Update Q-values using SARSA(λ) with eligibility traces.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state reached
            done: Whether episode ended
        """
        if done:
            target = reward
        else:
            next_action = self.act(next_state, training=True)
            target = reward + self.gamma * self.Q[next_state][next_action]
        
        # TD error
        delta = target - self.Q[state][action]
        
        # Update eligibility trace for current state-action
        self.eligibility_traces[state][action] += 1
        
        # Update all Q-values and eligibility traces
        for s in list(self.Q.keys()):
            for a in range(self.n_actions):
                self.Q[s][a] += self.alpha * delta * self.eligibility_traces[s][a]
                self.eligibility_traces[s][a] *= self.gamma * self.lambda_trace
        
        if done:
            # Reset eligibility traces for new episode
            self.eligibility_traces.clear()
            self.decay_epsilon()

    def get_hyperparameters(self) -> dict:
        """Get hyperparameters including lambda."""
        params = super().get_hyperparameters()
        params['lambda_trace'] = self.lambda_trace
        return params