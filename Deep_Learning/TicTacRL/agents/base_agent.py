"""
Base agent interface for TicTacRL project.
Defines common methods for all RL agents.
"""

from abc import ABC, abstractmethod
from typing import Tuple, List, Optional
import numpy as np


class BaseAgent(ABC):
    """
    Base class for all RL agents.
    """
    
    def __init__(self, player_id: int):
        """
        Initialize the agent.
        
        Args:
            player_id: Player ID (1 for X, 2 for O)
        """
        self.player_id = player_id
        self.training_mode = True
    
    @abstractmethod
    def select_action(self, board: np.ndarray, valid_moves: List[Tuple[int, int]]) -> Tuple[int, int]:
        """
        Select an action given the current board state.
        
        Args:
            board: Current board state
            valid_moves: List of valid moves
            
        Returns:
            Selected action as (row, col) tuple
        """
        pass
    
    @abstractmethod
    def update(self, state: str, action: Tuple[int, int], reward: float, 
               next_state: str, done: bool) -> None:
        """
        Update the agent's policy based on experience.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        pass
    
    @abstractmethod
    def save(self, filepath: str) -> None:
        """
        Save the agent's learned parameters.
        
        Args:
            filepath: Path to save the agent
        """
        pass
    
    @abstractmethod
    def load(self, filepath: str) -> None:
        """
        Load the agent's learned parameters.
        
        Args:
            filepath: Path to load the agent from
        """
        pass
    
    def set_training_mode(self, training: bool) -> None:
        """
        Set whether the agent is in training mode.
        
        Args:
            training: True for training mode, False for evaluation mode
        """
        self.training_mode = training
    
    def get_player_id(self) -> int:
        """
        Get the player ID.
        
        Returns:
            Player ID (1 for X, 2 for O)
        """
        return self.player_id 