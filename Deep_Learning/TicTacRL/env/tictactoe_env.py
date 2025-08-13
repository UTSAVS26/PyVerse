"""
TicTacToe environment for reinforcement learning.
Provides a gym-like interface for RL agents.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from utils.state_utils import (
    check_winner, is_game_over, get_valid_moves, 
    get_reward, print_board, encode_board_state
)


class TicTacToeEnv:
    """
    TicTacToe environment for reinforcement learning.
    """
    
    def __init__(self):
        """Initialize the environment."""
        self.board = None
        self.current_player = None
        self.game_history = []
        self.reset()
    
    def reset(self) -> np.ndarray:
        """
        Reset the environment to initial state.
        
        Returns:
            Initial board state
        """
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1  # X starts
        self.game_history = []
        return self.board.copy()
    
    def step(self, action: Tuple[int, int]) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: (row, col) tuple representing the move
            
        Returns:
            (observation, reward, done, info) tuple
        """
        row, col = action
        
        # Validate action
        if not self._is_valid_action(action):
            raise ValueError(f"Invalid action: {action}")
        
        # Make the move
        self.board[row, col] = self.current_player
        self.game_history.append((action, self.current_player))
        
        # Check if game is over
        winner = check_winner(self.board)
        done = winner is not None
        
        # Calculate reward
        reward = get_reward(winner, self.current_player)
        
        # Switch players if game continues
        if not done:
            self.current_player = 3 - self.current_player  # Switch between 1 and 2
        
        info = {
            'winner': winner,
            'current_player': self.current_player,
            'valid_moves': get_valid_moves(self.board) if not done else []
        }
        
        return self.board.copy(), reward, done, info
    
    def _is_valid_action(self, action: Tuple[int, int]) -> bool:
        """
        Check if an action is valid.
        
        Args:
            action: (row, col) tuple
            
        Returns:
            True if action is valid, False otherwise
        """
        row, col = action
        return (0 <= row < 3 and 0 <= col < 3 and 
                self.board[row, col] == 0)
    
    def get_valid_actions(self) -> list:
        """
        Get list of valid actions.
        
        Returns:
            List of (row, col) tuples for valid moves
        """
        return get_valid_moves(self.board)
    
    def get_state(self) -> str:
        """
        Get current state as string for Q-table lookup.
        
        Returns:
            String representation of current board state
        """
        return encode_board_state(self.board)
    
    def render(self) -> None:
        """
        Render the current board state.
        """
        print_board(self.board)
        if self.current_player == 1:
            print("Current player: X")
        else:
            print("Current player: O")
    
    def get_observation(self) -> np.ndarray:
        """
        Get current observation (board state).
        
        Returns:
            Copy of current board state
        """
        return self.board.copy()
    
    def is_terminal(self) -> bool:
        """
        Check if the game is in a terminal state.
        
        Returns:
            True if game is over, False otherwise
        """
        return is_game_over(self.board)
    
    def get_winner(self) -> Optional[int]:
        """
        Get the winner of the game.
        
        Returns:
            1 if X wins, 2 if O wins, 0 if draw, None if game continues
        """
        return check_winner(self.board)
    
    def get_game_history(self) -> list:
        """
        Get the history of moves made in the current game.
        
        Returns:
            List of (action, player) tuples
        """
        return self.game_history.copy() 