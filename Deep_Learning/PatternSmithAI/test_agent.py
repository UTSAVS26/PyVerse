"""
Tests for PatternSmithAI Agent Module
Tests AI agent, neural network, and pattern evaluation functionality.
"""

import pytest
import numpy as np
import os
import tempfile
import torch
import json

from canvas import PatternCanvas, ColorPalette
from patterns import PatternFactory
from agent import (
    PatternState, PatternAction, PatternNeuralNetwork, 
    PatternAgent, PatternEvaluator
)


class TestPatternState:
    """Test cases for PatternState class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.canvas = PatternCanvas(400, 400, "white")
        self.state = PatternState(self.canvas)
    
    def test_state_initialization(self):
        """Test pattern state initialization."""
        assert self.state.canvas == self.canvas
        assert hasattr(self.state, 'pixel_data')
        assert hasattr(self.state, 'features')
    
    def test_pixel_data_shape(self):
        """Test pixel data shape."""
        assert self.state.pixel_data.shape == (400, 400, 3)
    
    def test_features_shape(self):
        """Test features shape."""
        # Features should be 64*64 + 3 = 4099
        assert self.state.features.shape == (4099,)
        assert self.state.features.dtype == np.float64
    
    def test_features_normalization(self):
        """Test that features are properly normalized."""
        # Pixel values should be normalized to [0, 1]
        pixel_features = self.state.features[:-3]
        assert np.all(pixel_features >= 0)
        assert np.all(pixel_features <= 1)
    
    def test_symmetry_calculation(self):
        """Test symmetry calculation."""
        # Empty canvas should have low symmetry
        symmetry = self.state._calculate_symmetry()
        assert 0 <= symmetry <= 1
        
        # Draw a symmetric pattern
        self.canvas.draw_circle(200, 200, 50, "red", "blue")
        new_state = PatternState(self.canvas)
        new_symmetry = new_state._calculate_symmetry()
        assert 0 <= new_symmetry <= 1
    
    def test_complexity_calculation(self):
        """Test complexity calculation."""
        # Empty canvas should have low complexity
        complexity = self.state._calculate_complexity()
        assert 0 <= complexity <= 1
        
        # Draw a complex pattern
        self.canvas.draw_circle(200, 200, 50, "red", "blue")
        new_state = PatternState(self.canvas)
        new_complexity = new_state._calculate_complexity()
        assert 0 <= new_complexity <= 1
    
    def test_color_variety_calculation(self):
        """Test color variety calculation."""
        # Empty canvas should have low color variety
        color_variety = self.state._calculate_color_variety()
        assert 0 <= color_variety <= 1
        
        # Draw a colorful pattern
        self.canvas.draw_circle(200, 200, 50, "red", "blue")
        new_state = PatternState(self.canvas)
        new_color_variety = new_state._calculate_color_variety()
        assert 0 <= new_color_variety <= 1


class TestPatternAction:
    """Test cases for PatternAction class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.action = PatternAction("geometric", {"pattern_type": "circles", "count": 10})
    
    def test_action_initialization(self):
        """Test action initialization."""
        assert self.action.action_type == "geometric"
        assert self.action.parameters == {"pattern_type": "circles", "count": 10}
    
    def test_action_to_dict(self):
        """Test action to dictionary conversion."""
        action_dict = self.action.to_dict()
        assert action_dict["action_type"] == "geometric"
        assert action_dict["parameters"] == {"pattern_type": "circles", "count": 10}
    
    def test_action_from_dict(self):
        """Test action creation from dictionary."""
        action_dict = {
            "action_type": "mandala",
            "parameters": {"layers": 5, "base_shape": "circle"}
        }
        action = PatternAction.from_dict(action_dict)
        assert action.action_type == "mandala"
        assert action.parameters == {"layers": 5, "base_shape": "circle"}


class TestPatternNeuralNetwork:
    """Test cases for PatternNeuralNetwork class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.network = PatternNeuralNetwork(input_size=100, hidden_size=64, output_size=32)
    
    def test_network_initialization(self):
        """Test neural network initialization."""
        assert isinstance(self.network, PatternNeuralNetwork)
        assert hasattr(self.network, 'fc1')
        assert hasattr(self.network, 'fc2')
        assert hasattr(self.network, 'fc3')
        assert hasattr(self.network, 'dropout')
    
    def test_network_forward_pass(self):
        """Test neural network forward pass."""
        # Create dummy input
        x = torch.randn(1, 100)
        
        # Forward pass
        output = self.network(x)
        
        # Check output shape
        assert output.shape == (1, 32)
        assert not torch.isnan(output).any()
    
    def test_network_with_different_input_sizes(self):
        """Test network with different input sizes."""
        input_sizes = [50, 100, 200]
        
        for input_size in input_sizes:
            network = PatternNeuralNetwork(input_size=input_size, hidden_size=64, output_size=32)
            x = torch.randn(1, input_size)
            output = network(x)
            assert output.shape == (1, 32)


class TestPatternAgent:
    """Test cases for PatternAgent class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.canvas = PatternCanvas(400, 400, "white")
        self.color_palette = ColorPalette()
        self.agent = PatternAgent(self.canvas, self.color_palette)
    
    def test_agent_initialization(self):
        """Test agent initialization."""
        assert self.agent.canvas == self.canvas
        assert self.agent.color_palette == self.color_palette
        assert hasattr(self.agent, 'policy_net')
        assert hasattr(self.agent, 'target_net')
        assert hasattr(self.agent, 'memory')
        assert hasattr(self.agent, 'action_space')
    
    def test_action_space_creation(self):
        """Test action space creation."""
        assert len(self.agent.action_space) > 0
        assert all(isinstance(action, PatternAction) for action in self.agent.action_space)
    
    def test_get_state(self):
        """Test getting current state."""
        state = self.agent.get_state()
        assert isinstance(state, PatternState)
        assert state.canvas == self.canvas
    
    def test_select_action(self):
        """Test action selection."""
        state = self.agent.get_state()
        action = self.agent.select_action(state)
        assert isinstance(action, PatternAction)
        assert action in self.agent.action_space
    
    def test_execute_action(self):
        """Test action execution."""
        state = self.agent.get_state()
        action = self.agent.action_space[0]  # Use first action
        
        next_state, reward = self.agent.execute_action(action)
        
        assert isinstance(next_state, PatternState)
        assert isinstance(reward, float)
        assert next_state.canvas == self.canvas
    
    def test_reward_calculation(self):
        """Test reward calculation."""
        state = self.agent.get_state()
        reward = self.agent._calculate_reward(state)
        assert isinstance(reward, float)
        assert reward >= 0  # Reward should be non-negative
    
    def test_memory_operations(self):
        """Test memory operations."""
        state = self.agent.get_state()
        action = self.agent.action_space[0]
        next_state = self.agent.get_state()
        reward = 1.0
        done = False
        
        # Remember experience
        self.agent.remember(state, action, reward, next_state, done)
        
        # Check memory size
        assert len(self.agent.memory) > 0
    
    def test_replay_training(self):
        """Test replay training."""
        # Add some experiences to memory
        for _ in range(50):
            state = self.agent.get_state()
            action = self.agent.action_space[0]
            next_state = self.agent.get_state()
            reward = 1.0
            done = False
            self.agent.remember(state, action, reward, next_state, done)
        
        # Test replay
        self.agent.replay()
        # Should not raise an exception
        assert True
    
    def test_target_network_update(self):
        """Test target network update."""
        # Store original weights
        original_weights = self.agent.target_net.fc1.weight.clone()
        
        # Update target network
        self.agent.update_target_network()
        
        # Check that weights were updated
        new_weights = self.agent.target_net.fc1.weight.clone()
        assert torch.allclose(original_weights, new_weights)
    
    def test_model_save_load(self):
        """Test model saving and loading."""
        temp_dir = tempfile.mkdtemp()
        model_path = os.path.join(temp_dir, "test_model.pth")
        
        try:
            # Save model
            self.agent.save_model(model_path)
            assert os.path.exists(model_path)
            
            # Load model
            new_agent = PatternAgent(self.canvas, self.color_palette)
            new_agent.load_model(model_path)
            
            # Check that models are the same
            for param1, param2 in zip(self.agent.policy_net.parameters(), 
                                    new_agent.policy_net.parameters()):
                assert torch.allclose(param1, param2)
                
        finally:
            # Clean up
            if os.path.exists(model_path):
                os.remove(model_path)
            os.rmdir(temp_dir)
    
    def test_generate_pattern(self):
        """Test pattern generation."""
        result = self.agent.generate_pattern(steps=5)
        assert result == self.canvas
        
        # Check that something was drawn
        pixel_data = self.canvas.get_pixel_data()
        non_white_pixels = np.any(pixel_data != [255, 255, 255], axis=2)
        assert np.any(non_white_pixels)
    
    def test_get_training_stats(self):
        """Test getting training statistics."""
        stats = self.agent.get_training_stats()
        assert isinstance(stats, dict)
        assert 'epsilon' in stats
        assert 'memory_size' in stats
        assert 'training_history' in stats


class TestPatternEvaluator:
    """Test cases for PatternEvaluator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.canvas = PatternCanvas(400, 400, "white")
        self.evaluator = PatternEvaluator()
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        assert hasattr(self.evaluator, 'evaluation_criteria')
        assert len(self.evaluator.evaluation_criteria) == 4
    
    def test_evaluate_empty_pattern(self):
        """Test evaluation of empty pattern."""
        scores = self.evaluator.evaluate_pattern(self.canvas)
        
        assert isinstance(scores, dict)
        assert 'symmetry' in scores
        assert 'complexity' in scores
        assert 'color_harmony' in scores
        assert 'balance' in scores
        
        # All scores should be between 0 and 1
        for score in scores.values():
            assert 0 <= score <= 1
    
    def test_evaluate_pattern_with_content(self):
        """Test evaluation of pattern with content."""
        # Draw something on canvas
        self.canvas.draw_circle(200, 200, 50, "red", "blue")
        
        scores = self.evaluator.evaluate_pattern(self.canvas)
        
        assert isinstance(scores, dict)
        assert 'symmetry' in scores
        assert 'complexity' in scores
        assert 'color_harmony' in scores
        assert 'balance' in scores
        
        # All scores should be between 0 and 1
        for score in scores.values():
            assert 0 <= score <= 1
    
    def test_calculate_balance(self):
        """Test balance calculation."""
        balance = self.evaluator._calculate_balance(self.canvas)
        assert 0 <= balance <= 1
        
        # Draw something off-center
        self.canvas.draw_circle(100, 100, 30, "red", "blue")
        new_balance = self.evaluator._calculate_balance(self.canvas)
        assert 0 <= new_balance <= 1
    
    def test_get_overall_score(self):
        """Test overall score calculation."""
        scores = {
            'symmetry': 0.5,
            'complexity': 0.3,
            'color_harmony': 0.7,
            'balance': 0.6
        }
        
        overall_score = self.evaluator.get_overall_score(scores)
        assert isinstance(overall_score, float)
        assert 0 <= overall_score <= 1
        
        # Test with different score combinations
        scores2 = {
            'symmetry': 1.0,
            'complexity': 1.0,
            'color_harmony': 1.0,
            'balance': 1.0
        }
        overall_score2 = self.evaluator.get_overall_score(scores2)
        assert overall_score2 == 1.0
    
    def test_evaluation_consistency(self):
        """Test evaluation consistency."""
        # Evaluate same pattern multiple times
        scores1 = self.evaluator.evaluate_pattern(self.canvas)
        scores2 = self.evaluator.evaluate_pattern(self.canvas)
        
        # Scores should be the same for identical patterns
        for key in scores1:
            assert abs(scores1[key] - scores2[key]) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__])
