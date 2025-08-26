from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Dict, Any

class BaseAgent(ABC):
    """
    Abstract base class for all reinforcement learning agents
    
    This class defines the interface that all agents must implement
    """
    
    def __init__(self, state_size: int, action_size: int, **kwargs):
        """
        Initialize the agent
        
        Args:
            state_size: Dimension of the state space
            action_size: Number of possible actions
            **kwargs: Additional agent-specific parameters
        """
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = kwargs.get('epsilon', 1.0)  # Exploration rate
        self.epsilon_min = kwargs.get('epsilon_min', 0.01)
        self.epsilon_decay = kwargs.get('epsilon_decay', 0.995)
        
    @abstractmethod
    def act(self, state: np.ndarray) -> int:
        """
        Choose an action based on the current state
        
        Args:
            state: Current state observation
            
        Returns:
            Action to take (integer)
        """
        pass
    
    @abstractmethod
    def train(self, state: np.ndarray, action: int, reward: float, 
              next_state: np.ndarray, done: bool) -> None:
        """
        Train the agent on a single experience
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is finished
        """
        pass
    
    def update_epsilon(self) -> None:
        """Update exploration rate (epsilon-greedy)"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, filepath: str) -> None:
        """
        Save the agent's model/parameters
        
        Args:
            filepath: Path to save the model
        """
        pass
    
    def load(self, filepath: str) -> None:
        """
        Load the agent's model/parameters
        
        Args:
            filepath: Path to load the model from
        """
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Get agent information and parameters"""
        return {
            'type': self.__class__.__name__,
            'state_size': self.state_size,
            'action_size': self.action_size,
            'epsilon': self.epsilon,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay
        }
