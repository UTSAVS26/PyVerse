"""
Monte Carlo agent for TicTacToe.
Implements Monte Carlo control with first-visit updates.
"""

import numpy as np
import random
import pickle
from typing import Tuple, List, Dict
from agents.base_agent import BaseAgent
from utils.state_utils import encode_board_state


class MonteCarloAgent(BaseAgent):
    """
    Monte Carlo agent for TicTacToe.
    """
    
    def __init__(self, player_id: int, learning_rate: float = 0.1, epsilon: float = 0.1):
        """
        Initialize Monte Carlo agent.
        
        Args:
            player_id: Player ID (1 for X, 2 for O)
            learning_rate: Learning rate for value updates
            epsilon: Exploration rate for epsilon-greedy policy
        """
        super().__init__(player_id)
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.q_table = {}  # State-action value table
        self.returns = {}  # Returns for each state-action pair
        self.visit_counts = {}  # Visit counts for each state-action pair
    
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
        Store experience for Monte Carlo update (called at end of episode).
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        # This method is called during episode execution
        # Actual updates happen in update_episode method
        pass
    
    def update_episode(self, episode_history: List[Tuple[str, Tuple[int, int], float]]) -> None:
        """
        Update Q-values using Monte Carlo method.
        
        Args:
            episode_history: List of (state, action, reward) tuples for the episode
        """
        # Calculate returns for each step
        returns = []
        cumulative_return = 0
        
        # Calculate returns from the end of the episode
        for state, action, reward in reversed(episode_history):
            cumulative_return += reward
            returns.insert(0, cumulative_return)
        
        # Update Q-values using first-visit Monte Carlo
        visited_pairs = set()
        
        for i, (state, action, _) in enumerate(episode_history):
            state_action_pair = (state, action)
            
            # First-visit: only update if this state-action pair hasn't been visited
            if state_action_pair not in visited_pairs:
                visited_pairs.add(state_action_pair)
                
                # Initialize if needed
                if state not in self.q_table:
                    self.q_table[state] = {}
                if action not in self.q_table[state]:
                    self.q_table[state][action] = 0.0
                if state_action_pair not in self.returns:
                    self.returns[state_action_pair] = []
                if state_action_pair not in self.visit_counts:
                    self.visit_counts[state_action_pair] = 0
                
                # Add return to the list
                self.returns[state_action_pair].append(returns[i])
                self.visit_counts[state_action_pair] += 1
                
                # Update Q-value using average of all returns
                avg_return = np.mean(self.returns[state_action_pair])
                self.q_table[state][action] = avg_return
    
    def save(self, filepath: str) -> None:
        """
        Save Q-table and returns to file.
        
        Args:
            filepath: Path to save the agent
        """
        data = {
            'q_table': self.q_table,
            'returns': self.returns,
            'visit_counts': self.visit_counts
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filepath: str) -> None:
        """
        Load Q-table and returns from file.
        
        Args:
            filepath: Path to load the agent from
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.q_table = data['q_table']
            self.returns = data['returns']
            self.visit_counts = data['visit_counts']
    
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