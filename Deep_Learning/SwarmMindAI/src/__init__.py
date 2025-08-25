"""
SwarmMindAI - Advanced Multi-Agent Swarm Intelligence Framework

A cutting-edge simulation framework for autonomous multi-agent swarm coordination,
featuring advanced reinforcement learning algorithms and emergent behavior analysis.
"""

__version__ = "1.0.0"
__author__ = "SwarmMindAI Team"
__email__ = "contact@swarmmindai.com"

from .environment import SwarmEnvironment
from .agents import HeterogeneousSwarm, BaseAgent
from .algorithms import MultiAgentPPO, MultiAgentDQN
from .communication import CommunicationProtocol
from .visualization import SwarmVisualizer

__all__ = [
    "SwarmEnvironment",
    "HeterogeneousSwarm", 
    "BaseAgent",
    "MultiAgentPPO",
    "MultiAgentDQN",
    "CommunicationProtocol",
    "SwarmVisualizer"
]
