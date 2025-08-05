"""
Utility modules for visualization, data processing, and helper functions.
"""

from pathlib import Path
from .viz import render_env, get_renderer, WarehouseRenderer


def ensure_outputs_directory():
    """
    Ensure outputs directory structure exists.
    Creates directories programmatically when needed.
    """
    project_root = Path(__file__).parent.parent.parent.parent
    
    directories = [
        'outputs',
        'outputs/runs', 
        'outputs/models',
        'outputs/validation_results',
        'outputs/comparison_results'
    ]
    
    for dir_path in directories:
        (project_root / dir_path).mkdir(parents=True, exist_ok=True)


def get_outputs_dir():
    """Get outputs directory, creating if needed."""
    project_root = Path(__file__).parent.parent.parent.parent
    outputs_dir = project_root / 'outputs'
    outputs_dir.mkdir(exist_ok=True)
    return outputs_dir


__all__ = [
    "render_env",
    "get_renderer", 
    "WarehouseRenderer",
    "ensure_outputs_directory",
    "get_outputs_dir",
]