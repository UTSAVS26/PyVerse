"""
Tests for the train module
"""

import pytest
import numpy as np
import tempfile
import os
import json
from unittest.mock import patch, MagicMock

# Import the modules to test
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import ArtForgeTrainer


class TestArtForgeTrainer:
    """Test cases for the ArtForgeTrainer class"""
    
    def test_trainer_initialization(self):
        """Test trainer initialization"""
        trainer = ArtForgeTrainer(
            canvas_width=400,
            canvas_height=300,
            max_strokes=25,
            save_dir="test_gallery"
        )
        
        assert trainer.canvas_width == 400
        assert trainer.canvas_height == 300
        assert trainer.max_strokes == 25
        assert trainer.save_dir == "test_gallery"
        assert len(trainer.episode_rewards) == 0
        assert len(trainer.episode_lengths) == 0
        assert len(trainer.episode_coverages) == 0
        assert len(trainer.episode_color_diversities) == 0
        assert len(trainer.training_losses) == 0
        assert trainer.best_reward == float('-inf')
        assert trainer.best_coverage == 0.0
    
    def test_trainer_directory_creation(self):
        """Test that trainer creates necessary directories"""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_dir = os.path.join(temp_dir, "test_gallery")
            
            # Create trainer (this should create directories)
            trainer = ArtForgeTrainer(save_dir=save_dir)
            
            # Force directory creation by accessing save_dir
            _ = trainer.save_dir
            
            # Check that directories were created
            assert os.path.exists(save_dir)
            assert os.path.exists(os.path.join(save_dir, "artworks"))
            assert os.path.exists(os.path.join(save_dir, "models"))
    
    def test_train_episode_basic(self):
        """Test basic episode training"""
        trainer = ArtForgeTrainer(
            canvas_width=100,
            canvas_height=100,
            max_strokes=5,
            save_dir="test_gallery"
        )
        
        # Train one episode
        episode_stats = trainer.train_episode(render=False, save_artwork=False)
        
        assert isinstance(episode_stats, dict)
        assert 'episode' in episode_stats
        assert 'reward' in episode_stats
        assert 'length' in episode_stats
        assert 'coverage' in episode_stats
        assert 'color_diversity' in episode_stats
        assert 'stroke_types' in episode_stats
        
        # Check that statistics were updated
        assert len(trainer.episode_rewards) == 1
        assert len(trainer.episode_lengths) == 1
        assert len(trainer.episode_coverages) == 1
        assert len(trainer.episode_color_diversities) == 1
    
    def test_train_episode_with_save(self):
        """Test episode training with artwork saving"""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_dir = os.path.join(temp_dir, "test_gallery")
            
            trainer = ArtForgeTrainer(
                canvas_width=100,
                canvas_height=100,
                max_strokes=3,
                save_dir=save_dir
            )
            
            # Train episode with saving
            episode_stats = trainer.train_episode(render=False, save_artwork=True)
            
            # Check that artwork was saved
            artworks_dir = os.path.join(save_dir, "artworks")
            artwork_files = [f for f in os.listdir(artworks_dir) if f.endswith('.png')]
            assert len(artwork_files) > 0
    
    def test_train_episode_with_render(self):
        """Test episode training with rendering (mocked)"""
        with patch('matplotlib.pyplot.show') as mock_show:
            with patch('matplotlib.pyplot.pause') as mock_pause:
                trainer = ArtForgeTrainer(
                    canvas_width=50,
                    canvas_height=50,
                    max_strokes=3,
                    save_dir="test_gallery"
                )
                
                # Train episode with rendering
                episode_stats = trainer.train_episode(render=True, save_artwork=False)
                
                # Check that rendering was called
                assert mock_pause.called
    
    def test_multiple_episodes(self):
        """Test training multiple episodes"""
        trainer = ArtForgeTrainer(
            canvas_width=100,
            canvas_height=100,
            max_strokes=3,
            save_dir="test_gallery"
        )
        
        # Train multiple episodes
        for i in range(3):
            episode_stats = trainer.train_episode(render=False, save_artwork=False)
            assert episode_stats['episode'] == i + 1
        
        # Check that all statistics were recorded
        assert len(trainer.episode_rewards) == 3
        assert len(trainer.episode_lengths) == 3
        assert len(trainer.episode_coverages) == 3
        assert len(trainer.episode_color_diversities) == 3
    
    def test_best_performance_tracking(self):
        """Test tracking of best performance"""
        trainer = ArtForgeTrainer(
            canvas_width=100,
            canvas_height=100,
            max_strokes=3,
            save_dir="test_gallery"
        )
        
        # Train a few episodes
        for i in range(3):
            episode_stats = trainer.train_episode(render=False, save_artwork=False)
        
        # Check that best performance was tracked
        assert trainer.best_reward > float('-inf')
        assert trainer.best_coverage > 0.0
        
        # Check that best models were saved
        models_dir = os.path.join(trainer.save_dir, "models")
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
        assert len(model_files) > 0
    
    def test_save_intermediate_states(self):
        """Test saving intermediate states"""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_dir = os.path.join(temp_dir, "test_gallery")
            
            trainer = ArtForgeTrainer(
                canvas_width=100,
                canvas_height=100,
                max_strokes=25,
                save_dir=save_dir
            )
            
            # Train episode with intermediate states
            episode_stats = trainer.train_episode(render=False, save_artwork=True)
            
            # Check that intermediate states were saved
            artworks_dir = os.path.join(save_dir, "artworks")
            step_files = [f for f in os.listdir(artworks_dir) if 'step_' in f]
            assert len(step_files) > 0
    
    def test_save_checkpoint(self):
        """Test saving checkpoints"""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_dir = os.path.join(temp_dir, "test_gallery")
            
            trainer = ArtForgeTrainer(
                canvas_width=100,
                canvas_height=100,
                max_strokes=3,
                save_dir=save_dir
            )
            
            # Train an episode
            trainer.train_episode(render=False, save_artwork=False)
            
            # Save checkpoint
            trainer._save_checkpoint(episode=1)
            
            # Check that checkpoint was saved
            models_dir = os.path.join(save_dir, "models")
            checkpoint_files = [f for f in os.listdir(models_dir) if 'checkpoint' in f]
            stats_files = [f for f in os.listdir(models_dir) if 'training_stats' in f]
            
            assert len(checkpoint_files) > 0
            assert len(stats_files) > 0
    
    def test_load_checkpoint(self):
        """Test loading checkpoints"""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_dir = os.path.join(temp_dir, "test_gallery")
            
            trainer = ArtForgeTrainer(
                canvas_width=100,
                canvas_height=100,
                max_strokes=3,
                save_dir=save_dir
            )
            
            # Train an episode and save checkpoint
            trainer.train_episode(render=False, save_artwork=False)
            trainer._save_checkpoint(episode=1)
            
            # Load checkpoint
            checkpoint_path = os.path.join(save_dir, "models", "checkpoint_episode_1.pth")
            trainer.load_checkpoint(checkpoint_path)
            
            # Verify that checkpoint was loaded (this would require more detailed testing
            # of the actual model parameters, but we can at least check the method runs)
            assert os.path.exists(checkpoint_path)
    
    def test_generate_artwork(self):
        """Test artwork generation"""
        trainer = ArtForgeTrainer(
            canvas_width=100,
            canvas_height=100,
            max_strokes=5,
            save_dir="test_gallery"
        )
        
        # Train a few episodes first
        for i in range(2):
            trainer.train_episode(render=False, save_artwork=False)
        
        # Generate artwork
        artwork = trainer.generate_artwork(num_strokes=3, render=False)
        
        assert isinstance(artwork, np.ndarray)
        assert artwork.shape == (100, 100, 3)
        assert artwork.dtype == np.uint8
    
    def test_generate_artwork_with_render(self):
        """Test artwork generation with rendering (mocked)"""
        with patch('matplotlib.pyplot.show') as mock_show:
            trainer = ArtForgeTrainer(
                canvas_width=100,
                canvas_height=100,
                max_strokes=5,
                save_dir="test_gallery"
            )
            
            # Train a few episodes first
            for i in range(2):
                trainer.train_episode(render=False, save_artwork=False)
            
            # Generate artwork with rendering
            artwork = trainer.generate_artwork(num_strokes=3, render=True)
            
            # Check that rendering was called
            assert mock_show.called
    
    def test_plot_training_progress(self):
        """Test plotting training progress (mocked)"""
        with patch('matplotlib.pyplot.savefig') as mock_savefig:
            with patch('matplotlib.pyplot.show') as mock_show:
                trainer = ArtForgeTrainer(
                    canvas_width=100,
                    canvas_height=100,
                    max_strokes=3,
                    save_dir="test_gallery"
                )
                
                # Train a few episodes to get some data
                for i in range(3):
                    trainer.train_episode(render=False, save_artwork=False)
                
                # Plot training progress
                trainer._plot_training_progress()
                
                # Check that plot was saved and shown
                assert mock_savefig.called
                assert mock_show.called
    
    def test_training_statistics_consistency(self):
        """Test that training statistics are consistent"""
        trainer = ArtForgeTrainer(
            canvas_width=100,
            canvas_height=100,
            max_strokes=3,
            save_dir="test_gallery"
        )
        
        # Train multiple episodes
        for i in range(3):
            episode_stats = trainer.train_episode(render=False, save_artwork=False)
            
            # Check that episode stats match recorded stats
            assert trainer.episode_rewards[-1] == episode_stats['reward']
            assert trainer.episode_lengths[-1] == episode_stats['length']
            assert trainer.episode_coverages[-1] == episode_stats['coverage']
            assert trainer.episode_color_diversities[-1] == episode_stats['color_diversity']
    
    def test_episode_statistics_validation(self):
        """Test that episode statistics are valid"""
        trainer = ArtForgeTrainer(
            canvas_width=100,
            canvas_height=100,
            max_strokes=3,
            save_dir="test_gallery"
        )
        
        episode_stats = trainer.train_episode(render=False, save_artwork=False)
        
        # Validate statistics
        assert episode_stats['reward'] >= 0.0  # Should be non-negative
        assert 1 <= episode_stats['length'] <= 3  # Should be within max_strokes
        assert 0.0 <= episode_stats['coverage'] <= 1.0  # Should be percentage
        assert 0.0 <= episode_stats['color_diversity'] <= 1.0  # Should be percentage
        assert isinstance(episode_stats['stroke_types'], list)
    
    def test_training_losses_tracking(self):
        """Test that training losses are tracked"""
        trainer = ArtForgeTrainer(
            canvas_width=100,
            canvas_height=100,
            max_strokes=10,
            save_dir="test_gallery"
        )
        
        # Train multiple episodes to accumulate enough memory for training
        for i in range(10):
            trainer.train_episode(render=False, save_artwork=False)
        
        # Check that training losses were recorded
        assert len(trainer.training_losses) > 0
        
        # Check that losses have the expected structure
        for loss_info in trainer.training_losses:
            assert 'critic_loss' in loss_info
            assert 'actor_loss' in loss_info
            assert isinstance(loss_info['critic_loss'], float)
            assert isinstance(loss_info['actor_loss'], float)
    
    def test_save_best_model(self):
        """Test saving best models"""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_dir = os.path.join(temp_dir, "test_gallery")
            
            trainer = ArtForgeTrainer(
                canvas_width=100,
                canvas_height=100,
                max_strokes=3,
                save_dir=save_dir
            )
            
            # Manually trigger best model save without training
            trainer._save_best_model("test_model")
            
            # Check that model was saved
            models_dir = os.path.join(save_dir, "models")
            model_files = [f for f in os.listdir(models_dir) if 'test_model' in f]
            assert len(model_files) > 0
    
    def test_training_with_custom_parameters(self):
        """Test training with custom parameters"""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_dir = os.path.join(temp_dir, "test_gallery")
            
            trainer = ArtForgeTrainer(
                canvas_width=200,
                canvas_height=150,
                max_strokes=10,
                save_dir=save_dir
            )
            
            # Train with custom parameters
            trainer.train(
                num_episodes=3,
                render_frequency=2,
                save_frequency=1,
                checkpoint_frequency=2
            )
            
            # Check that training completed
            assert len(trainer.episode_rewards) == 3
            assert len(trainer.episode_lengths) == 3
            assert len(trainer.episode_coverages) == 3
            assert len(trainer.episode_color_diversities) == 3


class TestMainFunction:
    """Test the main function"""
    
    @patch('train.ArtForgeTrainer')
    def test_main_function(self, mock_trainer_class):
        """Test the main function execution"""
        # Mock the trainer
        mock_trainer = MagicMock()
        mock_trainer_class.return_value = mock_trainer
        
        # Mock the generate_artwork return value to be a numpy array
        mock_trainer.generate_artwork.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Mock the save_dir to be a valid path
        mock_trainer.save_dir = "test_gallery"
        
        # Import and run main
        from train import main
        main()
        
        # Check that trainer was created and used
        mock_trainer_class.assert_called_once()
        mock_trainer.train.assert_called_once()
        mock_trainer.generate_artwork.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
