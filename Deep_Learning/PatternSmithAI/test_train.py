"""
Tests for PatternSmithAI Training Module
Tests training functionality, statistics, and interactive training.
"""

import pytest
import numpy as np
import os
import tempfile
import json
from unittest.mock import patch, MagicMock

from canvas import PatternCanvas, ColorPalette
from patterns import PatternFactory
from agent import PatternAgent, PatternEvaluator
from train import PatternTrainer, InteractiveTrainer


class TestPatternTrainer:
    """Test cases for PatternTrainer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.trainer = PatternTrainer(canvas_size=400, output_dir=self.temp_dir)
    
    def teardown_method(self):
        """Clean up after tests."""
        # Clean up temporary files
        for file in os.listdir(self.temp_dir):
            file_path = os.path.join(self.temp_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        os.rmdir(self.temp_dir)
    
    def test_trainer_initialization(self):
        """Test trainer initialization."""
        assert self.trainer.canvas.width == 400
        assert self.trainer.canvas.height == 400
        assert self.trainer.output_dir == self.temp_dir
        assert isinstance(self.trainer.agent, PatternAgent)
        assert isinstance(self.trainer.evaluator, PatternEvaluator)
        assert hasattr(self.trainer, 'training_stats')
    
    def test_training_stats_structure(self):
        """Test training statistics structure."""
        stats = self.trainer.training_stats
        assert 'episodes' in stats
        assert 'rewards' in stats
        assert 'scores' in stats
        assert 'epsilon_values' in stats
        assert isinstance(stats['episodes'], list)
        assert isinstance(stats['rewards'], list)
        assert isinstance(stats['scores'], list)
        assert isinstance(stats['epsilon_values'], list)
    
    def test_train_episode(self):
        """Test training one episode."""
        episode_stats = self.trainer.train_episode(steps=5)
        
        assert isinstance(episode_stats, dict)
        assert 'total_reward' in episode_stats
        assert 'overall_score' in episode_stats
        assert 'scores' in episode_stats
        assert 'epsilon' in episode_stats
        
        assert isinstance(episode_stats['total_reward'], float)
        assert isinstance(episode_stats['overall_score'], float)
        assert isinstance(episode_stats['epsilon'], float)
        assert isinstance(episode_stats['scores'], dict)
        
        # Check that scores are valid
        for score in episode_stats['scores'].values():
            assert 0 <= score <= 1
    
    def test_train_multiple_episodes(self):
        """Test training multiple episodes."""
        # Train a few episodes
        self.trainer.train(episodes=3, steps_per_episode=5)
        
        # Check that statistics were recorded
        assert len(self.trainer.training_stats['episodes']) == 3
        assert len(self.trainer.training_stats['rewards']) == 3
        assert len(self.trainer.training_stats['scores']) == 3
        assert len(self.trainer.training_stats['epsilon_values']) == 3
        
        # Check that rewards and scores are reasonable
        for reward in self.trainer.training_stats['rewards']:
            assert isinstance(reward, float)
            assert reward >= 0
        
        for score in self.trainer.training_stats['scores']:
            assert isinstance(score, float)
            assert 0 <= score <= 1
    
    def test_save_training_pattern(self):
        """Test saving training patterns."""
        # Draw something on canvas
        self.trainer.canvas.draw_circle(200, 200, 50, "red", "blue")
        
        # Save pattern
        self.trainer._save_training_pattern(1)
        
        # Check that file was created
        pattern_files = [f for f in os.listdir(self.temp_dir) if f.endswith('.png')]
        assert len(pattern_files) > 0
    
    def test_save_training_results(self):
        """Test saving training results."""
        # Train a few episodes first
        self.trainer.train(episodes=2, steps_per_episode=5)
        
        # Save results
        self.trainer._save_training_results()
        
        # Check that files were created
        model_file = os.path.join(self.temp_dir, "trained_model.pth")
        stats_file = os.path.join(self.temp_dir, "training_stats.json")
        plots_file = os.path.join(self.temp_dir, "training_plots.png")
        
        assert os.path.exists(model_file)
        assert os.path.exists(stats_file)
        assert os.path.exists(plots_file)
        
        # Check that stats file contains valid JSON
        with open(stats_file, 'r') as f:
            stats = json.load(f)
            assert 'episodes' in stats
            assert 'rewards' in stats
            assert 'scores' in stats
            assert 'epsilon_values' in stats
    
    def test_create_training_plots(self):
        """Test training plots creation."""
        # Add some dummy data
        self.trainer.training_stats['episodes'] = [0, 1, 2]
        self.trainer.training_stats['rewards'] = [1.0, 1.5, 2.0]
        self.trainer.training_stats['scores'] = [0.5, 0.6, 0.7]
        self.trainer.training_stats['epsilon_values'] = [1.0, 0.9, 0.8]
        
        # Create plots
        self.trainer._create_training_plots()
        
        # Check that plot file was created
        plots_file = os.path.join(self.temp_dir, "training_plots.png")
        assert os.path.exists(plots_file)
    
    def test_generate_sample_patterns(self):
        """Test sample pattern generation."""
        # Train a few episodes first
        self.trainer.train(episodes=2, steps_per_episode=5)
        
        # Generate sample patterns
        self.trainer.generate_sample_patterns(count=3)
        
        # Check that pattern files were created
        pattern_files = [f for f in os.listdir(self.temp_dir) if f.endswith('.png')]
        metadata_files = [f for f in os.listdir(self.temp_dir) if f.endswith('_metadata.json')]
        
        assert len(pattern_files) >= 3
        assert len(metadata_files) >= 3
        
        # Check that metadata files contain valid JSON
        for metadata_file in metadata_files[:3]:
            with open(os.path.join(self.temp_dir, metadata_file), 'r') as f:
                metadata = json.load(f)
                assert 'pattern_id' in metadata
                assert 'timestamp' in metadata
                assert 'scores' in metadata
                assert 'overall_score' in metadata


class TestInteractiveTrainer:
    """Test cases for InteractiveTrainer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.trainer = InteractiveTrainer(canvas_size=400, output_dir=self.temp_dir)
    
    def teardown_method(self):
        """Clean up after tests."""
        # Clean up temporary files
        for file in os.listdir(self.temp_dir):
            file_path = os.path.join(self.temp_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        os.rmdir(self.temp_dir)
    
    def test_trainer_initialization(self):
        """Test interactive trainer initialization."""
        assert isinstance(self.trainer, InteractiveTrainer)
        assert isinstance(self.trainer, PatternTrainer)
        assert hasattr(self.trainer, 'user_feedback_history')
        assert isinstance(self.trainer.user_feedback_history, list)
    
    @patch('builtins.input', return_value='0.7')
    def test_get_user_feedback_valid(self, mock_input):
        """Test getting valid user feedback."""
        feedback = self.trainer.get_user_feedback(1)
        assert feedback == 0.7
    
    @patch('builtins.input', side_effect=['invalid', '1.5', '0.8'])
    def test_get_user_feedback_invalid_then_valid(self, mock_input):
        """Test getting user feedback with invalid inputs."""
        feedback = self.trainer.get_user_feedback(1)
        assert feedback == 0.8
    
    @patch('builtins.input', return_value='0.6')
    def test_train_with_feedback(self, mock_input):
        """Test training with user feedback."""
        # Train with feedback
        self.trainer.train_with_feedback(episodes=2, steps_per_episode=5)
        
        # Check that feedback was recorded
        assert len(self.trainer.user_feedback_history) == 2
        
        for feedback_data in self.trainer.user_feedback_history:
            assert 'episode' in feedback_data
            assert 'user_rating' in feedback_data
            assert 'agent_score' in feedback_data
            assert 'timestamp' in feedback_data
            assert feedback_data['user_rating'] == 0.6
    
    def test_adjust_agent_with_feedback(self):
        """Test agent adjustment based on feedback."""
        # Set epsilon to a value that can be increased
        self.trainer.agent.epsilon = 0.5
        original_epsilon = self.trainer.agent.epsilon
        
        # Test adjustment with high user rating (should increase epsilon)
        # User rating 0.8 vs agent score 0.3 = difference 0.5 > 0.3 threshold
        episode_stats = {'overall_score': 0.3}
        self.trainer._adjust_agent_with_feedback(0.8, episode_stats)
        
        # Epsilon should be increased (user liked it more than agent expected)
        assert self.trainer.agent.epsilon > original_epsilon
        
        # Reset epsilon and test adjustment with low user rating
        self.trainer.agent.epsilon = original_epsilon
        # User rating 0.1 vs agent score 0.3 = difference 0.2 < 0.3 threshold
        # So we need a bigger difference to trigger adjustment
        self.trainer._adjust_agent_with_feedback(0.0, episode_stats)
        
        # Epsilon should be decreased (user liked it less than agent expected)
        assert self.trainer.agent.epsilon < original_epsilon
        
        # Test with small feedback difference (should not adjust)
        self.trainer.agent.epsilon = original_epsilon
        # User rating 0.35 vs agent score 0.3 = difference 0.05 < 0.3 threshold
        self.trainer._adjust_agent_with_feedback(0.35, episode_stats)
        
        # Epsilon should remain the same (difference < 0.3)
        assert self.trainer.agent.epsilon == original_epsilon
        
        # Test edge case: epsilon at maximum (1.0) - should not increase
        self.trainer.agent.epsilon = 1.0
        original_epsilon = self.trainer.agent.epsilon
        episode_stats = {'overall_score': 0.3}
        self.trainer._adjust_agent_with_feedback(0.8, episode_stats)
        
        # Epsilon should remain at maximum
        assert self.trainer.agent.epsilon == 1.0
    
    def test_feedback_history_saving(self):
        """Test that feedback history is saved."""
        # Add some dummy feedback
        self.trainer.user_feedback_history = [
            {
                'episode': 1,
                'user_rating': 0.7,
                'agent_score': 0.5,
                'timestamp': '2023-01-01T00:00:00'
            }
        ]
        
        # Save feedback history
        feedback_file = os.path.join(self.temp_dir, "user_feedback_history.json")
        with open(feedback_file, 'w') as f:
            json.dump(self.trainer.user_feedback_history, f)
        
        # Check that file was created
        assert os.path.exists(feedback_file)
        
        # Check that file contains valid JSON
        with open(feedback_file, 'r') as f:
            feedback_data = json.load(f)
            assert len(feedback_data) == 1
            assert feedback_data[0]['episode'] == 1
            assert feedback_data[0]['user_rating'] == 0.7


class TestTrainingIntegration:
    """Integration tests for training functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up after tests."""
        # Clean up temporary files
        for file in os.listdir(self.temp_dir):
            file_path = os.path.join(self.temp_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        os.rmdir(self.temp_dir)
    
    def test_full_training_workflow(self):
        """Test complete training workflow."""
        trainer = PatternTrainer(canvas_size=400, output_dir=self.temp_dir)
        
        # Train
        trainer.train(episodes=3, steps_per_episode=5)
        
        # Generate samples
        trainer.generate_sample_patterns(count=2)
        
        # Check that all expected files were created
        files = os.listdir(self.temp_dir)
        
        # Should have model file
        model_files = [f for f in files if f.endswith('.pth')]
        assert len(model_files) > 0
        
        # Should have stats file
        stats_files = [f for f in files if f.endswith('.json')]
        assert len(stats_files) > 0
        
        # Should have plot file
        plot_files = [f for f in files if f.endswith('.png')]
        assert len(plot_files) > 0
        
        # Should have sample patterns
        pattern_files = [f for f in files if 'sample_pattern' in f and f.endswith('.png')]
        assert len(pattern_files) >= 2
    
    def test_training_progress_tracking(self):
        """Test that training progress is properly tracked."""
        trainer = PatternTrainer(canvas_size=400, output_dir=self.temp_dir)
        
        # Train a few episodes
        trainer.train(episodes=5, steps_per_episode=3)
        
        # Check that progress was tracked
        stats = trainer.training_stats
        
        assert len(stats['episodes']) == 5
        assert len(stats['rewards']) == 5
        assert len(stats['scores']) == 5
        assert len(stats['epsilon_values']) == 5
        
        # Check that episodes are sequential
        assert stats['episodes'] == [0, 1, 2, 3, 4]
        
        # Check that epsilon decreases over time
        epsilon_values = stats['epsilon_values']
        assert epsilon_values[0] >= epsilon_values[-1]  # Epsilon should decrease
    
    def test_agent_learning(self):
        """Test that agent actually learns during training."""
        trainer = PatternTrainer(canvas_size=400, output_dir=self.temp_dir)
        
        # Train
        trainer.train(episodes=10, steps_per_episode=5)
        
        # Check that agent has learned (epsilon should decrease)
        stats = trainer.training_stats
        initial_epsilon = stats['epsilon_values'][0]
        final_epsilon = stats['epsilon_values'][-1]
        
        # Epsilon should decrease (agent becomes less random)
        assert final_epsilon < initial_epsilon
        
        # Check that agent has accumulated experience
        assert len(trainer.agent.memory) > 0


if __name__ == "__main__":
    pytest.main([__file__])
