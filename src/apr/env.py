from __future__ import annotations
import numpy as np
from typing import Tuple, Dict

class WarehouseEnv:
    """
    Minimal grid-world for parcel delivery.
    Symbols:
        A  – agent
        P  – pick-up (start)
        D  – drop-off (goal)
        #  – obstacle
    """

    ACTIONS: Dict[int, Tuple[int, int]] = {
        0: (-1, 0),   # up
        1: (1, 0),    # down
        2: (0, -1),   # left
        3: (0, 1),    # right
    }

    def __init__(
        self,
        size: Tuple[int, int] = (6, 6),
        obstacles: int = 3,
        max_steps: int = 100,
        seed: int | None = None,
    ):
        self.rng = np.random.default_rng(seed)
        self.n_rows, self.n_cols = size
        self.n_actions = 4
        self.n_obstacles = obstacles
        self.max_steps = max_steps

        # public “spaces” so agents know the dims
        self.action_space = self.n_actions
        self.observation_space = (self.n_rows, self.n_cols)

        # placeholders filled in reset()
        self.pickup = (0, 0)
        self.dropoff = (self.n_rows - 1, self.n_cols - 1)
        self.agent_pos = self.pickup
        self.obstacles = set()
        self.steps = 0

    # --------------------------------------------------------------------- #
    # Core gym-like API                                                     #
    # --------------------------------------------------------------------- #

    def reset(self) -> Tuple[int, int]:
        """Start a new episode and return the initial state (row, col)."""
        self.steps = 0
        self.agent_pos = self.pickup

        # sample obstacle cells (never on P or D)
        occupied = {self.pickup, self.dropoff}
        self.obstacles.clear()
        while len(self.obstacles) < self.n_obstacles:
            cell = (self.rng.integers(0, self.n_rows),
                    self.rng.integers(0, self.n_cols))
            if cell not in occupied:
                self.obstacles.add(cell)
                occupied.add(cell)
        return self.agent_pos

    def step(self, action: int):
        """Do one action; return (next_state, reward, done, info_dict)."""
        self.steps += 1
        dr, dc = WarehouseEnv.ACTIONS[action]
        r, c = self.agent_pos
        nr, nc = r + dr, c + dc

        reward = -1                      # living penalty to encourage speed
        done = False

        # valid move?
        if (
            0 <= nr < self.n_rows
            and 0 <= nc < self.n_cols
            and (nr, nc) not in self.obstacles
        ):
            self.agent_pos = (nr, nc)
        else:                            # bumped wall / obstacle
            reward -= 4

        # reached goal?
        if self.agent_pos == self.dropoff:
            reward += 10
            done = True

        # time limit?
        if self.steps >= self.max_steps:
            done = True

        return self.agent_pos, reward, done, {}

    # ------------------------------------------------------------------ #
    # Pretty rendering so you can watch the agent move in the console    #
    # ------------------------------------------------------------------ #

    def render(self, mode: str = "human"):
        grid = np.full((self.n_rows, self.n_cols), " ")
        for (r, c) in self.obstacles:
            grid[r, c] = "#"
        pr, pc = self.pickup
        dr, dc = self.dropoff
        ar, ac = self.agent_pos
        grid[pr, pc] = "P"
        grid[dr, dc] = "D"
        grid[ar, ac] = "A"

        border = "+" + "-" * self.n_cols + "+"
        print(border)
        for row in grid:
            print("|" + "".join(row) + "|")
        print(border)