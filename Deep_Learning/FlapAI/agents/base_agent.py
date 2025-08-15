from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import numpy as np

class BaseAgent(ABC):
    """Base class for all AI agents in FlapAI."""
    
    def __init__(self, name: str = "BaseAgent"):
        self.name = name
        self.training = True
        self.episode_count = 0
        self.total_reward = 0
        self.best_score = 0
        
    @abstractmethod
    def get_action(self, state: Dict[str, Any]) -> int:
        """
        Get the action to take given the current state.
        
        Args:
            state: Current game state dictionary
            
        Returns:
            action: 0 for no flap, 1 for flap
        """
        pass
        
    @abstractmethod
    def update(self, state: Dict[str, Any], action: int, 
               reward: float, next_state: Dict[str, Any], done: bool) -> None:
        """
        Update the agent with experience.
        
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
        """Save the agent to a file."""
        pass
        
    @abstractmethod
    def load(self, filepath: str) -> None:
        """Load the agent from a file."""
        pass
        
    def reset_episode(self) -> None:
        """Reset episode-specific variables."""
        self.episode_count += 1
        self.total_reward = 0
        
    def get_stats(self) -> Dict[str, Any]:
        """Get current agent statistics."""
        return {
            'name': self.name,
            'episode_count': self.episode_count,
            'best_score': self.best_score,
            'training': self.training
        }
        
    def set_training(self, training: bool) -> None:
        """Set training mode."""
        self.training = training
        
    def update_best_score(self, score: int) -> None:
        """Update the best score achieved."""
        if score > self.best_score:
            self.best_score = score

class RandomAgent(BaseAgent):
    """Random agent for baseline comparison."""
    
    def __init__(self):
        super().__init__("RandomAgent")
        
    def get_action(self, state: Dict[str, Any]) -> int:
        """Return random action."""
        return np.random.randint(0, 2)
        
    def update(self, state: Dict[str, Any], action: int, 
               reward: float, next_state: Dict[str, Any], done: bool) -> None:
        """Random agent doesn't learn."""
        pass
        
    def save(self, filepath: str) -> None:
        """Random agent has nothing to save."""
        pass
        
    def load(self, filepath: str) -> None:
        """Random agent has nothing to load."""
        pass

class HumanAgent(BaseAgent):
    """Human agent for manual play."""
    
    def __init__(self):
        super().__init__("HumanAgent")
        self.last_action = 0
        
    def get_action(self, state: Dict[str, Any]) -> int:
        """Return the last action (should be set externally)."""
        return self.last_action
        
    def set_action(self, action: int) -> None:
        """Set the action for the next step."""
        self.last_action = action
        
    def update(self, state: Dict[str, Any], action: int, 
               reward: float, next_state: Dict[str, Any], done: bool) -> None:
        """Human agent doesn't learn."""
        pass
        
    def save(self, filepath: str) -> None:
        """Human agent has nothing to save."""
        pass
        
    def load(self, filepath: str) -> None:
        """Human agent has nothing to load."""
        pass 