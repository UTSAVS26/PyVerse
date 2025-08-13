"""
Test cases for TicTacRL project.
Tests environment, agents, and utilities.
"""

import pytest
import numpy as np
import tempfile
import os
from env.tictactoe_env import TicTacToeEnv
from agents.q_learning_agent import QLearningAgent
from agents.monte_carlo_agent import MonteCarloAgent
from utils.state_utils import (
    encode_board_state, decode_board_state, get_valid_moves,
    get_linear_position, get_2d_position, check_winner,
    is_game_over, get_reward, print_board
)


class TestStateUtils:
    """Test cases for state utilities."""
    
    def test_encode_decode_board_state(self):
        """Test board state encoding and decoding."""
        # Test empty board
        board = np.zeros((3, 3), dtype=int)
        state_str = encode_board_state(board)
        decoded_board = decode_board_state(state_str)
        assert np.array_equal(board, decoded_board)
        
        # Test board with moves
        board[0, 0] = 1  # X
        board[1, 1] = 2  # O
        state_str = encode_board_state(board)
        decoded_board = decode_board_state(state_str)
        assert np.array_equal(board, decoded_board)
    
    def test_get_valid_moves(self):
        """Test getting valid moves."""
        board = np.zeros((3, 3), dtype=int)
        valid_moves = get_valid_moves(board)
        assert len(valid_moves) == 9
        
        # Add some moves
        board[0, 0] = 1
        board[1, 1] = 2
        valid_moves = get_valid_moves(board)
        assert len(valid_moves) == 7
        assert (0, 0) not in valid_moves
        assert (1, 1) not in valid_moves
    
    def test_position_conversion(self):
        """Test 2D to linear position conversion."""
        # Test all positions
        for i in range(3):
            for j in range(3):
                linear_pos = get_linear_position(i, j)
                row, col = get_2d_position(linear_pos)
                assert row == i
                assert col == j
    
    def test_check_winner(self):
        """Test winner checking."""
        # Test no winner
        board = np.zeros((3, 3), dtype=int)
        assert check_winner(board) is None
        
        # Test X wins in first row
        board[0, 0] = board[0, 1] = board[0, 2] = 1
        assert check_winner(board) == 1
        
        # Test O wins in diagonal
        board = np.zeros((3, 3), dtype=int)
        board[0, 2] = board[1, 1] = board[2, 0] = 2
        assert check_winner(board) == 2
        
        # Test draw
        board = np.array([[1, 2, 1], [2, 1, 2], [2, 1, 2]], dtype=int)
        assert check_winner(board) == 0
    
    def test_is_game_over(self):
        """Test game over checking."""
        board = np.zeros((3, 3), dtype=int)
        assert not is_game_over(board)
        
        # Add winning condition
        board[0, 0] = board[0, 1] = board[0, 2] = 1
        assert is_game_over(board)
    
    def test_get_reward(self):
        """Test reward calculation."""
        # Win
        assert get_reward(1, 1) == 1.0
        assert get_reward(2, 2) == 1.0
        
        # Loss
        assert get_reward(2, 1) == -1.0
        assert get_reward(1, 2) == -1.0
        
        # Draw
        assert get_reward(0, 1) == 0.5
        assert get_reward(0, 2) == 0.5
        
        # Game continues
        assert get_reward(None, 1) == 0.0


class TestTicTacToeEnv:
    """Test cases for TicTacToe environment."""
    
    def setup_method(self):
        """Set up test environment."""
        self.env = TicTacToeEnv()
    
    def test_reset(self):
        """Test environment reset."""
        # Make some moves
        self.env.step((0, 0))
        self.env.step((1, 1))
        
        # Reset
        board = self.env.reset()
        assert np.array_equal(board, np.zeros((3, 3), dtype=int))
        assert self.env.current_player == 1
    
    def test_valid_action(self):
        """Test valid action validation."""
        assert self.env._is_valid_action((0, 0))
        assert not self.env._is_valid_action((3, 0))  # Out of bounds
        assert not self.env._is_valid_action((0, 3))  # Out of bounds
        
        # Make a move
        self.env.step((0, 0))
        assert not self.env._is_valid_action((0, 0))  # Already taken
    
    def test_step(self):
        """Test environment step."""
        # Valid move
        board, reward, done, info = self.env.step((0, 0))
        assert board[0, 0] == 1
        assert self.env.current_player == 2
        assert not done
        
        # Invalid move should raise error
        with pytest.raises(ValueError):
            self.env.step((0, 0))  # Already taken
    
    def test_winning_game(self):
        """Test winning game scenario."""
        # X wins in first row
        self.env.step((0, 0))  # X
        self.env.step((1, 0))  # O
        self.env.step((0, 1))  # X
        self.env.step((1, 1))  # O
        board, reward, done, info = self.env.step((0, 2))  # X wins
        
        assert done
        assert info['winner'] == 1
        assert reward == 1.0
    
    def test_draw_game(self):
        """Test draw game scenario."""
        # Play a draw game
        moves = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 0), (1, 2), (2, 1), (2, 0), (2, 2)]
        
        for i, move in enumerate(moves):
            board, reward, done, info = self.env.step(move)
            if i < len(moves) - 1:  # Not the last move
                assert not done
            else:  # Last move
                assert done
                assert info['winner'] == 0
    
    def test_get_valid_actions(self):
        """Test getting valid actions."""
        actions = self.env.get_valid_actions()
        assert len(actions) == 9
        
        # Make a move
        self.env.step((0, 0))
        actions = self.env.get_valid_actions()
        assert len(actions) == 8
        assert (0, 0) not in actions


class TestQLearningAgent:
    """Test cases for Q-learning agent."""
    
    def setup_method(self):
        """Set up test agent."""
        self.agent = QLearningAgent(player_id=1)
    
    def test_initialization(self):
        """Test agent initialization."""
        assert self.agent.player_id == 1
        assert self.agent.epsilon == 0.1
        assert self.agent.q_table == {}
    
    def test_select_action(self):
        """Test action selection."""
        board = np.zeros((3, 3), dtype=int)
        valid_moves = [(0, 0), (0, 1), (1, 1)]
        
        # In training mode with exploration
        self.agent.set_training_mode(True)
        action = self.agent.select_action(board, valid_moves)
        assert action in valid_moves
        
        # In evaluation mode (no exploration)
        self.agent.set_training_mode(False)
        action = self.agent.select_action(board, valid_moves)
        assert action in valid_moves
    
    def test_update(self):
        """Test Q-value updates."""
        state = "000000000"
        action = (0, 0)
        reward = 1.0
        next_state = "100000000"
        
        self.agent.update(state, action, reward, next_state, True)
        
        assert state in self.agent.q_table
        assert action in self.agent.q_table[state]
        assert self.agent.q_table[state][action] > 0
    
    def test_save_load(self):
        """Test saving and loading agent."""
        # Train the agent a bit
        state = "000000000"
        action = (0, 0)
        self.agent.update(state, action, 1.0, "100000000", True)
        
        # Save
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_file = f.name
        
        try:
            self.agent.save(temp_file)
            
            # Load into new agent
            new_agent = QLearningAgent(player_id=1)
            new_agent.load(temp_file)
            
            assert new_agent.q_table == self.agent.q_table
        finally:
            os.unlink(temp_file)
    
    def test_epsilon_decay(self):
        """Test epsilon decay."""
        original_epsilon = self.agent.get_epsilon()
        self.agent.set_epsilon(0.05)
        assert self.agent.get_epsilon() == 0.05


class TestMonteCarloAgent:
    """Test cases for Monte Carlo agent."""
    
    def setup_method(self):
        """Set up test agent."""
        self.agent = MonteCarloAgent(player_id=1)
    
    def test_initialization(self):
        """Test agent initialization."""
        assert self.agent.player_id == 1
        assert self.agent.epsilon == 0.1
        assert self.agent.q_table == {}
        assert self.agent.returns == {}
        assert self.agent.visit_counts == {}
    
    def test_select_action(self):
        """Test action selection."""
        board = np.zeros((3, 3), dtype=int)
        valid_moves = [(0, 0), (0, 1), (1, 1)]
        
        action = self.agent.select_action(board, valid_moves)
        assert action in valid_moves
    
    def test_update_episode(self):
        """Test episodic updates."""
        episode_history = [
            ("000000000", (0, 0), 0.0),
            ("100000000", (1, 1), 0.0),
            ("100020000", (0, 1), 1.0)  # Win
        ]
        
        self.agent.update_episode(episode_history)
        
        # Check that Q-values were updated
        assert len(self.agent.q_table) > 0
        assert len(self.agent.returns) > 0
    
    def test_save_load(self):
        """Test saving and loading agent."""
        # Train the agent a bit
        episode_history = [
            ("000000000", (0, 0), 1.0)
        ]
        self.agent.update_episode(episode_history)
        
        # Save
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_file = f.name
        
        try:
            self.agent.save(temp_file)
            
            # Load into new agent
            new_agent = MonteCarloAgent(player_id=1)
            new_agent.load(temp_file)
            
            assert new_agent.q_table == self.agent.q_table
            assert new_agent.returns == self.agent.returns
            assert new_agent.visit_counts == self.agent.visit_counts
        finally:
            os.unlink(temp_file)


class TestIntegration:
    """Integration tests."""
    
    def test_agent_vs_agent(self):
        """Test two agents playing against each other."""
        env = TicTacToeEnv()
        agent1 = QLearningAgent(player_id=1)
        agent2 = QLearningAgent(player_id=2)
        
        # Play a game
        env.reset()
        while not env.is_terminal():
            current_agent = agent1 if env.current_player == 1 else agent2
            valid_moves = env.get_valid_actions()
            
            if not valid_moves:
                break
            
            action = current_agent.select_action(env.get_observation(), valid_moves)
            env.step(action)
        
        # Game should end
        assert env.is_terminal()
        winner = env.get_winner()
        assert winner in [0, 1, 2]  # Draw, X wins, or O wins
    
    def test_environment_consistency(self):
        """Test environment consistency."""
        env = TicTacToeEnv()
        
        # Play a complete game
        moves = [(0, 0), (1, 1), (0, 1), (1, 0), (0, 2)]  # X wins
        
        for move in moves:
            board, reward, done, info = env.step(move)
            assert board.shape == (3, 3)
            assert reward in [-1.0, 0.0, 0.5, 1.0]
        
        assert env.is_terminal()
        assert env.get_winner() == 1  # X wins


if __name__ == "__main__":
    pytest.main([__file__]) 