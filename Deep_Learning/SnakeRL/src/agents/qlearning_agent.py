import numpy as np
import pickle
from typing import Dict, Any, Tuple
from .base_agent import BaseAgent

class QLearningAgent(BaseAgent):
    """
    Q-Learning agent with tabular Q-table
    
    Suitable for smaller state spaces where all states can be enumerated
    """
    
    def __init__(self, state_size: int, action_size: int, **kwargs):
        """
        Initialize Q-Learning agent
        
        Args:
            state_size: Dimension of the state space
            action_size: Number of possible actions
            **kwargs: Additional parameters including:
                - learning_rate: Learning rate for Q-updates
                - discount_factor: Discount factor for future rewards
        """
        super().__init__(state_size, action_size, **kwargs)
        
        self.learning_rate = kwargs.get('learning_rate', 0.1)
        self.discount_factor = kwargs.get('discount_factor', 0.95)
        
        # Q-table: dictionary mapping state-action pairs to Q-values
        # For continuous state spaces, we discretize the state
        self.q_table = {}
        
    def _discretize_state(self, state: np.ndarray) -> Tuple:
        """
        Discretize continuous state for tabular Q-learning
        
        Args:
            state: Continuous state array
            
        Returns:
            Discretized state tuple
        """
        # Discretize head and food positions (4x4 grid)
        head_x, head_y = int(state[0] * 4), int(state[1] * 4)
        food_x, food_y = int(state[2] * 4), int(state[3] * 4)
        
        # Direction is already discrete (0-3)
        direction = np.argmax(state[4:8])
        
        # Danger zones are already binary
        danger_left = int(state[8])
        danger_straight = int(state[9])
        danger_right = int(state[10])
        
        return (head_x, head_y, food_x, food_y, direction, 
                danger_left, danger_straight, danger_right)
    
    def act(self, state: np.ndarray) -> int:
        """
        Choose action using epsilon-greedy policy
        
        Args:
            state: Current state observation
            
        Returns:
            Action to take (0=left, 1=straight, 2=right)
        """
        # Epsilon-greedy exploration
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_size)
        
        # Greedy action selection
        discretized_state = self._discretize_state(state)
        q_values = self._get_q_values(discretized_state)
        return np.argmax(q_values)
    
    def _get_q_values(self, state: Tuple) -> np.ndarray:
        """
        Get Q-values for a given state
        
        Args:
            state: Discretized state tuple
            
        Returns:
            Array of Q-values for each action
        """
        q_values = np.zeros(self.action_size)
        for action in range(self.action_size):
            state_action = (state, action)
            if state_action in self.q_table:
                q_values[action] = self.q_table[state_action]
        return q_values
    
    def train(self, state: np.ndarray, action: int, reward: float, 
              next_state: np.ndarray, done: bool) -> None:
        """
        Update Q-table using Q-learning update rule
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is finished
        """
        current_state = self._discretize_state(state)
        next_state_disc = self._discretize_state(next_state)
        
        # Get current Q-value
        state_action = (current_state, action)
        if state_action not in self.q_table:
            self.q_table[state_action] = 0.0
        
        current_q = self.q_table[state_action]
        
        # Get max Q-value for next state
        next_q_values = self._get_q_values(next_state_disc)
        max_next_q = np.max(next_q_values) if not done else 0.0
        
        # Q-learning update rule
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[state_action] = new_q
    
    def save(self, filepath: str) -> None:
        """
        Save Q-table to file
        
        Args:
            filepath: Path to save the Q-table
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self.q_table, f)
    
    def load(self, filepath: str) -> None:
        """
        Load Q-table from file
        
        Args:
            filepath: Path to load the Q-table from
        """
        with open(filepath, 'rb') as f:
            self.q_table = pickle.load(f)
    
    def get_info(self) -> Dict[str, Any]:
        """Get agent information including Q-table size"""
        info = super().get_info()
        info.update({
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'q_table_size': len(self.q_table)
        })
        return info
