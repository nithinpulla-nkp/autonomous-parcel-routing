from __future__ import annotations
import numpy as np
from typing import Tuple, Dict, Set
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class WarehouseEnv:
    """
    Enhanced warehouse environment for autonomous parcel routing.
    
    Features:
    - Multiple packages to collect
    - Fixed shelf positions (obstacles)
    - Package pickup/delivery mechanics
    - Rich reward structure
    - Visual rendering support
    
    Symbols:
        A  – agent (robot)
        P  – package (to pick up)
        D  – drop-off (delivery zone)
        #  – shelf (obstacle)
    """

    ACTIONS: Dict[int, Tuple[int, int]] = {
        0: (-1, 0),   # up
        1: (1, 0),    # down
        2: (0, -1),   # left
        3: (0, 1),    # right
    }
    
    ACTION_NAMES: Dict[int, str] = {
        0: "up", 1: "down", 2: "left", 3: "right"
    }
    
    REVERSE_ACTIONS: Dict[str, int] = {
        "up": 0, "down": 1, "left": 2, "right": 3
    }

    def __init__(
        self,
        size: Tuple[int, int] = (6, 6),
        max_steps: int = 200,
        seed: int | None = None,
        layout: str = "default"
    ):
        self.rng = np.random.default_rng(seed)
        self.n_rows, self.n_cols = size
        self.n_actions = 4
        self.max_steps = max_steps
        self.layout = layout

        # Public "spaces" for agents
        self.action_space = self.n_actions
        self.observation_space = (self.n_rows, self.n_cols)

        # Initialize warehouse layout
        self._setup_warehouse_layout()
        
        # Episode state variables (reset in reset())
        self.agent_pos = self.robot_start
        self.packages_remaining = set()
        self.num_picked_up_items = 0
        self.carrying_packages = False
        self.steps = 0
        self.episode_complete = False

    def _setup_warehouse_layout(self):
        """Initialize warehouse layout with fixed positions."""
        if self.layout == "default":
            # Based on your notebook's warehouse layout
            self.shelf_positions = {
                (5, 4), (5, 0), (1, 0), (4, 1), (2, 5), (4, 4)
            }
            self.package_positions_initial = {
                (3, 2), (1, 2), (3, 5), (0, 1)
            }
            self.robot_start = (0, 0)
            self.dropoff_position = (5, 5)
        else:
            # Could add other layouts here
            self.shelf_positions = {(2, 2), (3, 3)}
            self.package_positions_initial = {(1, 1), (4, 4)}
            self.robot_start = (0, 0)
            self.dropoff_position = (5, 5)

    # --------------------------------------------------------------------- #
    # Core gym-like API                                                     #
    # --------------------------------------------------------------------- #

    def reset(self) -> Tuple[int, int]:
        """Start a new episode and return the initial state (row, col)."""
        self.steps = 0
        self.agent_pos = self.robot_start
        self.packages_remaining = self.package_positions_initial.copy()
        self.num_picked_up_items = 0
        self.carrying_packages = False
        self.episode_complete = False
        
        return self.agent_pos

    def step(self, action: int):
        """
        Execute one action in the warehouse environment.
        
        Returns:
            next_state: (row, col) position
            reward: immediate reward
            done: whether episode is complete
            info: additional info dict
        """
        self.steps += 1
        dr, dc = WarehouseEnv.ACTIONS[action]
        r, c = self.agent_pos
        nr, nc = r + dr, c + dc

        reward = 0
        done = False
        info = {
            'packages_remaining': len(self.packages_remaining),
            'packages_collected': self.num_picked_up_items,
            'carrying_packages': self.carrying_packages
        }

        # Check bounds
        if not (0 <= nr < self.n_rows and 0 <= nc < self.n_cols):
            reward -= 3  # Out of bounds penalty
            return self.agent_pos, reward, done, info

        # Check collision with shelves
        if (nr, nc) in self.shelf_positions:
            reward -= 20  # Heavy penalty for hitting shelves
            return self.agent_pos, reward, done, info

        # Valid move - update position
        self.agent_pos = (nr, nc)
        reward -= 1  # Step penalty to encourage efficiency

        # Check for package pickup
        if (nr, nc) in self.packages_remaining:
            self.packages_remaining.remove((nr, nc))
            self.num_picked_up_items += 1
            self.carrying_packages = True
            reward += 25  # Reward for picking up package
            info['action'] = 'pickup'

        # Check for delivery
        elif (nr, nc) == self.dropoff_position:
            if self.num_picked_up_items > 0:
                # Deliver all carried packages
                reward += 100 * self.num_picked_up_items  # Big reward for delivery
                info['delivered_packages'] = self.num_picked_up_items
                self.episode_complete = True
                done = True
                info['action'] = 'delivery'
            else:
                # Visited dropoff with no packages
                reward -= 5  # Small penalty
                info['action'] = 'empty_delivery'

        # Check episode termination
        if self.steps >= self.max_steps:
            done = True
            info['timeout'] = True

        return self.agent_pos, reward, done, info

    # ------------------------------------------------------------------ #
    # Rendering methods for visualization                                #
    # ------------------------------------------------------------------ #

    def render(self, mode: str = "human"):
        """Render the warehouse environment."""
        if mode == "human":
            self._render_console()
        elif mode == "matplotlib":
            self._render_matplotlib()
        elif mode == "sprites":
            self._render_sprites()
        else:
            raise ValueError(f"Unsupported render mode: {mode}")
    
    def _render_sprites(self):
        """Render using sprite images."""
        from apr.utils import render_env
        fig, ax = render_env(self)
        plt.show()
        return fig, ax

    def _render_console(self):
        """Simple console rendering with ASCII characters."""
        grid = np.full((self.n_rows, self.n_cols), " ")
        
        # Place shelves
        for (r, c) in self.shelf_positions:
            grid[r, c] = "#"
        
        # Place remaining packages
        for (r, c) in self.packages_remaining:
            grid[r, c] = "P"
        
        # Place dropoff
        dr, dc = self.dropoff_position
        grid[dr, dc] = "D"
        
        # Place agent (agent overwrites everything at its position)
        ar, ac = self.agent_pos
        if self.carrying_packages:
            grid[ar, ac] = "@"  # Agent carrying packages
        else:
            grid[ar, ac] = "A"  # Empty agent

        # Print grid
        border = "+" + "-" * self.n_cols + "+"
        print(border)
        for row in grid:
            print("|" + "".join(row) + "|")
        print(border)
        print(f"Step: {self.steps}, Packages: {self.num_picked_up_items}/{len(self.package_positions_initial)}")

    def _render_matplotlib(self, figsize=(8, 8)):
        """Rich matplotlib rendering with colors and symbols."""
        fig, ax = plt.subplots(figsize=figsize)
        
        # Set up grid
        ax.set_xlim(-0.5, self.n_cols - 0.5)
        ax.set_ylim(-0.5, self.n_rows - 0.5)
        ax.set_xticks(np.arange(-0.5, self.n_cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.n_rows, 1), minor=True)
        ax.grid(which="minor", color="black", linestyle="-", linewidth=0.5)
        ax.set_aspect("equal")
        ax.set_facecolor("white")

        # Draw shelves as black rectangles
        for (r, c) in self.shelf_positions:
            rect = patches.Rectangle(
                (c - 0.5, self.n_rows - r - 1.5), 1, 1, 
                color="black", label="Shelf" if (r, c) == next(iter(self.shelf_positions)) else ""
            )
            ax.add_patch(rect)

        # Draw remaining packages as blue circles
        for (r, c) in self.packages_remaining:
            circle = patches.Circle(
                (c, self.n_rows - r - 1), 0.3, 
                color="blue", label="Package" if (r, c) == next(iter(self.packages_remaining)) else ""
            )
            ax.add_patch(circle)

        # Draw dropoff as green square
        dr, dc = self.dropoff_position
        rect = patches.Rectangle(
            (dc - 0.4, self.n_rows - dr - 1.4), 0.8, 0.8,
            color="green", label="Dropoff"
        )
        ax.add_patch(rect)

        # Draw agent
        ar, ac = self.agent_pos
        agent_color = "red" if self.carrying_packages else "orange"
        agent_size = 0.4 if self.carrying_packages else 0.3
        circle = patches.Circle(
            (ac, self.n_rows - ar - 1), agent_size,
            color=agent_color, label=f"Agent ({'Carrying' if self.carrying_packages else 'Empty'})"
        )
        ax.add_patch(circle)

        # Labels and title
        ax.set_title(f"Warehouse Environment - Step {self.steps}")
        ax.set_xlabel("Column")
        ax.set_ylabel("Row (flipped)")
        ax.legend()
        
        # Remove ticks for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])
        
        plt.tight_layout()
        plt.show()

    def print_state(self):
        """Print detailed state information."""
        print(f"=== Warehouse State (Step {self.steps}) ===")
        print(f"Agent Position: {self.agent_pos}")
        print(f"Packages Collected: {self.num_picked_up_items}")
        print(f"Packages Remaining: {len(self.packages_remaining)}")
        print(f"Carrying Packages: {self.carrying_packages}")
        print(f"Episode Complete: {self.episode_complete}")
        if self.packages_remaining:
            print(f"Remaining Package Positions: {self.packages_remaining}")
        print("=" * 40)