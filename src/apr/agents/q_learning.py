import random, pickle
import numpy as np
from collections import defaultdict

class QLearningAgent:
    """Plain ε-greedy Q-Learning for discrete states & actions."""

    def __init__(
        self,
        obs_space,                  # unused but keeps signature generic
        act_space: int,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.2,
        epsilon_decay: float = 0.999,
        epsilon_min: float = 0.05,
    ):
        self.n_actions = act_space
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.Q = defaultdict(lambda: np.zeros(self.n_actions, dtype=float))

    # ------------------------------------------------------------------ #

    def act(self, state, training: bool = True) -> int:
        if training and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        return int(np.argmax(self.Q[state]))

    def learn(self, s, a, r, s2, done):
        best_next = np.max(self.Q[s2])
        target = r + (0 if done else self.gamma * best_next)
        self.Q[s][a] += self.alpha * (target - self.Q[s][a])

        if done:                                 # decay ε once per episode
            self.epsilon = max(self.epsilon * self.epsilon_decay,
                                self.epsilon_min)

    # ------------------------------------------------------------------ #
    # Persistence                                                        #
    # ------------------------------------------------------------------ #

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(dict(self.Q), f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            q_table = pickle.load(f)
        # infer action dim from first row
        n_actions = len(next(iter(q_table.values())))
        agent = cls(obs_space=None, act_space=n_actions)
        agent.Q = defaultdict(lambda: np.zeros(n_actions, dtype=float), q_table)
        return agent