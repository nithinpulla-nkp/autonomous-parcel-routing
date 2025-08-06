"""
Visualization utilities for warehouse environment rendering.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
from apr.resources import SPRITES_DIR

class WarehouseRenderer:
    """Rich matplotlib renderer with sprite images."""
    
    def __init__(self):
        self.sprites = self._load_sprites()
    
    def _load_sprites(self):
        """Load all sprite images."""
        sprites = {}
        sprite_files = {
            'robot': 'robot.png',
            'robot_carrying': 'robo2_package.png', 
            'package': 'package.png',
            'dropoff': 'icon-destination.png',
            'trap': 'trap.png'
        }
        
        for name, filename in sprite_files.items():
            try:
                sprites[name] = mpimg.imread(SPRITES_DIR / filename)
            except FileNotFoundError:
                print(f"Warning: Sprite {filename} not found, using fallback")
                sprites[name] = None
        
        return sprites
    
    def render_environment(self, env, figsize=(8, 8), title_suffix=""):
        """Render warehouse environment with sprites."""
        fig, ax = plt.subplots(figsize=figsize)
        
        # Grid setup
        ax.set_xlim(-0.5, env.n_cols - 0.5)
        ax.set_ylim(-0.5, env.n_rows - 0.5)
        ax.set_xticks(range(env.n_cols))
        ax.set_yticks(range(env.n_rows))
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")
        ax.set_facecolor("lightgray")
        
        # Draw shelves/traps
        for (r, c) in env.shelf_positions:
            if self.sprites['trap'] is not None:
                ax.imshow(
                    self.sprites['trap'], 
                    extent=(c - 0.5, c + 0.5, env.n_rows - r - 1.5, env.n_rows - r - 0.5)
                )
            else:
                # Fallback to black rectangle
                ax.add_patch(plt.Rectangle((c - 0.5, env.n_rows - r - 1.5), 1, 1, color='black'))
        
        # Draw packages
        for (r, c) in env.packages_remaining:
            if self.sprites['package'] is not None:
                ax.imshow(
                    self.sprites['package'],
                    extent=(c - 0.5, c + 0.5, env.n_rows - r - 1.5, env.n_rows - r - 0.5)
                )
            else:
                # Fallback to blue circle
                circle = plt.Circle((c, env.n_rows - r - 1), 0.3, color='blue')
                ax.add_patch(circle)
        
        # Draw dropoff
        dr, dc = env.dropoff_position
        if self.sprites['dropoff'] is not None:
            ax.imshow(
                self.sprites['dropoff'],
                extent=(dc - 0.5, dc + 0.5, env.n_rows - dr - 1.5, env.n_rows - dr - 0.5)
            )
        else:
            # Fallback to green square
            ax.add_patch(plt.Rectangle((dc - 0.4, env.n_rows - dr - 1.4), 0.8, 0.8, color='green'))
        
        # Draw robot
        ar, ac = env.agent_pos
        robot_sprite = 'robot_carrying' if env.carrying_packages else 'robot'
        
        if self.sprites[robot_sprite] is not None:
            ax.imshow(
                self.sprites[robot_sprite],
                extent=(ac - 0.5, ac + 0.5, env.n_rows - ar - 1.5, env.n_rows - ar - 0.5)
            )
        else:
            # Fallback to colored circle
            color = 'red' if env.carrying_packages else 'orange'
            circle = plt.Circle((ac, env.n_rows - ar - 1), 0.3, color=color)
            ax.add_patch(circle)
        
        # Labels and formatting
        ax.set_title(f"Warehouse Environment - Step {env.steps} {title_suffix}")
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        
        # Add status text
        status_text = f"Packages: {env.num_picked_up_items}/{len(env.package_positions_initial)} | "
        status_text += f"Carrying: {'Yes' if env.carrying_packages else 'No'}"
        ax.text(0.02, 0.98, status_text, transform=ax.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                verticalalignment='top')
        
        plt.tight_layout()
        return fig, ax

# Global renderer instance
_renderer = None

def get_renderer():
    """Get the global renderer instance."""
    global _renderer
    if _renderer is None:
        _renderer = WarehouseRenderer()
    return _renderer

def render_env(env, **kwargs):
    """Convenience function to render environment."""
    renderer = get_renderer()
    return renderer.render_environment(env, **kwargs)