"""
State utilities for TicTacRL project.
Handles encoding/decoding of board states for RL agents.
"""

import numpy as np
from typing import Tuple, List, Optional


def encode_board_state(board: np.ndarray) -> str:
    """
    Encode board state as a string for Q-table lookup.
    
    Args:
        board: 3x3 numpy array with 0 (empty), 1 (X), 2 (O)
    
    Returns:
        String representation of board state
    """
    return ''.join(str(cell) for cell in board.flatten())


def decode_board_state(state_str: str) -> np.ndarray:
    """
    Decode string state back to board array.
    
    Args:
        state_str: String representation of board state
    
    Returns:
        3x3 numpy array representing board
    """
    return np.array([int(c) for c in state_str]).reshape(3, 3)


def get_valid_moves(board: np.ndarray) -> List[Tuple[int, int]]:
    """
    Get list of valid moves (empty positions) on the board.
    
    Args:
        board: 3x3 numpy array
    
    Returns:
        List of (row, col) tuples for empty positions
    """
    valid_moves = []
    for i in range(3):
        for j in range(3):
            if board[i, j] == 0:
                valid_moves.append((i, j))
    return valid_moves


def get_linear_position(row: int, col: int) -> int:
    """
    Convert 2D position to linear position (0-8).
    
    Args:
        row: Row index (0-2)
        col: Column index (0-2)
    
    Returns:
        Linear position (0-8)
    """
    return row * 3 + col


def get_2d_position(linear_pos: int) -> Tuple[int, int]:
    """
    Convert linear position to 2D position.
    
    Args:
        linear_pos: Linear position (0-8)
    
    Returns:
        (row, col) tuple
    """
    return linear_pos // 3, linear_pos % 3


def check_winner(board: np.ndarray) -> Optional[int]:
    """
    Check if there's a winner on the board.
    
    Args:
        board: 3x3 numpy array
    
    Returns:
        1 if X wins, 2 if O wins, 0 if draw, None if game continues
    """
    # Check rows
    for i in range(3):
        if board[i, 0] == board[i, 1] == board[i, 2] != 0:
            return board[i, 0]
    
    # Check columns
    for j in range(3):
        if board[0, j] == board[1, j] == board[2, j] != 0:
            return board[0, j]
    
    # Check diagonals
    if board[0, 0] == board[1, 1] == board[2, 2] != 0:
        return board[0, 0]
    if board[0, 2] == board[1, 1] == board[2, 0] != 0:
        return board[0, 2]
    
    # Check for draw
    if 0 not in board:
        return 0
    
    # Game continues
    return None


def is_game_over(board: np.ndarray) -> bool:
    """
    Check if the game is over (win or draw).
    
    Args:
        board: 3x3 numpy array
    
    Returns:
        True if game is over, False otherwise
    """
    return check_winner(board) is not None


def get_reward(winner: Optional[int], player: int) -> float:
    """
    Get reward for a game outcome.
    
    Args:
        winner: Winner (1 for X, 2 for O, 0 for draw, None for ongoing)
        player: Player number (1 for X, 2 for O)
    
    Returns:
        Reward value
    """
    if winner is None:
        return 0.0  # Game continues
    elif winner == 0:
        return 0.5  # Draw
    elif winner == player:
        return 1.0  # Win
    else:
        return -1.0  # Loss


def print_board(board: np.ndarray) -> None:
    """
    Print board in a human-readable format.
    
    Args:
        board: 3x3 numpy array
    """
    symbols = {0: ' ', 1: 'X', 2: 'O'}
    
    print("  0 1 2")
    for i in range(3):
        row = f"{i} "
        for j in range(3):
            row += f"{symbols[board[i, j]]} "
        print(row)
    print() 