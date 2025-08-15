"""
Q-learning agent for TicTacToe.
Implements Q-learning with epsilon-greedy exploration.
"""

import numpy as np
import random
import pickle
from typing import Tuple, List, Dict
from agents.base_agent import BaseAgent
from utils.state_utils import encode_board_state


class QLearningAgent(BaseAgent):
    """
    Q-learning agent for TicTacToe.
    """
    
    def __init__(self, player_id: int, learning_rate: float = 0.1, 
                 discount_factor: float = 0.9, epsilon: float = 0.1):
        """
        Initialize Q-learning agent.
        
        Args:
            player_id: Player ID (1 for X, 2 for O)
            learning_rate: Learning rate for Q-value updates
            discount_factor: Discount factor for future rewards
            epsilon: Exploration rate for epsilon-greedy policy
        """
        super().__init__(player_id)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = {}  # State-action value table
    
    def select_action(self, board: np.ndarray, valid_moves: List[Tuple[int, int]]) -> Tuple[int, int]:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            board: Current board state
            valid_moves: List of valid moves
            
        Returns:
            Selected action as (row, col) tuple
        """
        state = encode_board_state(board)
        
        # Epsilon-greedy exploration
        if self.training_mode and random.random() < self.epsilon:
            return random.choice(valid_moves)
        
        # Greedy action selection
        return self._get_best_action(state, valid_moves)
    
    def _get_best_action(self, state: str, valid_moves: List[Tuple[int, int]]) -> Tuple[int, int]:
        """
        Get the best action according to Q-values.
        
        Args:
            state: Current state
            valid_moves: List of valid moves
            
        Returns:
            Best action as (row, col) tuple
        """
        if state not in self.q_table:
            # Initialize Q-values for this state
            self.q_table[state] = {}
        
        best_value = float('-inf')
        best_actions = []
        
        for action in valid_moves:
            if action not in self.q_table[state]:
                self.q_table[state][action] = 0.0
            
            value = self.q_table[state][action]
            if value > best_value:
                best_value = value
                best_actions = [action]
            elif value == best_value:
                best_actions.append(action)
        
        # If no Q-values or all equal, choose randomly
        if not best_actions:
            return random.choice(valid_moves)
        
        return random.choice(best_actions)
    
    def update(self, state: str, action: Tuple[int, int], reward: float, 
               next_state: str, done: bool) -> None:
        """
        Update Q-values using Q-learning update rule.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        if state not in self.q_table:
            self.q_table[state] = {}
        if action not in self.q_table[state]:
            self.q_table[state][action] = 0.0
        
        # Q-learning update rule
        if done:
            # Terminal state
            self.q_table[state][action] += self.learning_rate * (reward - self.q_table[state][action])
        else:
            # Non-terminal state
            if next_state not in self.q_table:
                self.q_table[next_state] = {}
            
            # Find max Q-value for next state
            max_next_q = 0.0
            if self.q_table[next_state]:
                max_next_q = max(self.q_table[next_state].values())
            
            # Q-learning update
            target = reward + self.discount_factor * max_next_q
            self.q_table[state][action] += self.learning_rate * (target - self.q_table[state][action])
    
    def save(self, filepath: str) -> None:
        """
        Save Q-table to file.
        
        Args:
            filepath: Path to save the Q-table
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self.q_table, f)
    
    def load(self, filepath: str) -> None:
        """
        Load Q-table from file.
        
        Args:
            filepath: Path to load the Q-table from
        """
        with open(filepath, 'rb') as f:
            self.q_table = pickle.load(f)
    
    def get_q_table_size(self) -> int:
        """
        Get the number of states in the Q-table.
        
        Returns:
            Number of states
        """
        return len(self.q_table)
    
    def get_total_q_values(self) -> int:
        """
        Get the total number of Q-values stored.
        
        Returns:
            Total number of Q-values
        """
        return sum(len(actions) for actions in self.q_table.values())
    
    def set_epsilon(self, epsilon: float) -> None:
        """
        Set the exploration rate.
        
        Args:
            epsilon: New exploration rate
        """
        self.epsilon = epsilon
    
    def get_epsilon(self) -> float:
        """
        Get the current exploration rate.
        
        Returns:
            Current epsilon value
        """
        return self.epsilon 