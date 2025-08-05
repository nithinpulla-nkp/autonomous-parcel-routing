"""
Packaged resources like sprites, configuration templates, and data files.
"""

from pathlib import Path

# Resource paths
SPRITES_DIR = Path(__file__).parent / "sprites"

__all__ = [
    "SPRITES_DIR",
]