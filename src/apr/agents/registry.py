"""
Agent registry for mapping algorithm names to agent classes.

This module provides a clean way to register and lookup agent classes,
replacing the fragile dynamic import system in train.py.
"""

from typing import Dict, Type
from .base import BaseAgent
from .q_learning import QLearningAgent
from .double_q_learning import DoubleQLearningAgent
from .sarsa import SarsaAgent, SarsaLambdaAgent


# Global registry mapping algorithm names to agent classes
AGENT_REGISTRY: Dict[str, Type[BaseAgent]] = {
    'q_learning': QLearningAgent,
    'qlearning': QLearningAgent,  # Alternative spelling
    'double_q_learning': DoubleQLearningAgent,
    'double_qlearning': DoubleQLearningAgent,  # Alternative spelling
    'sarsa': SarsaAgent,
    'sarsa_lambda': SarsaLambdaAgent,
}


def register_agent(name: str, agent_class: Type[BaseAgent]) -> None:
    """
    Register a new agent class in the registry.
    
    Args:
        name: Algorithm name (used in config files)
        agent_class: Agent class that inherits from BaseAgent
    """
    if not issubclass(agent_class, BaseAgent):
        raise ValueError(f"Agent class {agent_class} must inherit from BaseAgent")
    
    AGENT_REGISTRY[name.lower()] = agent_class


def get_agent_class(name: str) -> Type[BaseAgent]:
    """
    Get agent class by name.
    
    Args:
        name: Algorithm name
        
    Returns:
        Agent class
        
    Raises:
        ValueError: If algorithm name is not registered
    """
    name_lower = name.lower()
    if name_lower not in AGENT_REGISTRY:
        available = ', '.join(AGENT_REGISTRY.keys())
        raise ValueError(f"Unknown algorithm '{name}'. Available: {available}")
    
    return AGENT_REGISTRY[name_lower]


def list_available_agents() -> list:
    """
    Get list of all available agent names.
    
    Returns:
        List of registered algorithm names
    """
    return list(AGENT_REGISTRY.keys())


def create_agent(name: str, obs_space, act_space, **kwargs) -> BaseAgent:
    """
    Create an agent instance by name.
    
    Args:
        name: Algorithm name
        obs_space: Observation space
        act_space: Action space
        **kwargs: Additional agent parameters
        
    Returns:
        Agent instance
    """
    agent_class = get_agent_class(name)
    return agent_class(obs_space, act_space, **kwargs)