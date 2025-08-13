import sys
import os
import pytest
import numpy as np
import tempfile
import shutil

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.train_neat import NEATTrainer, train_neat
from training.train_dqn import DQNTrainer, train_dqn
from agents.neat_agent import NEATAgent
from agents.dqn_agent import DQNAgent
from game.flappy_bird import FlappyBirdGame

class TestNEATTrainer:
    """Test cases for the NEAT trainer."""
    
    def test_neat_trainer_initialization(self):
        """Test NEAT trainer initialization."""
        trainer = NEATTrainer(max_generations=10, render_training=False)
        
        assert trainer.max_generations == 10
        assert trainer.max_fitness == 1000
        assert trainer.render_training == False
        assert trainer.save_interval == 10
        assert trainer.game is not None
        assert trainer.population is not None
        assert trainer.best_agent is None
        assert trainer.best_score == 0
        
    def test_neat_trainer_evaluate_agent(self):
        """Test agent evaluation."""
        trainer = NEATTrainer(max_generations=1, render_training=False)
        agent = NEATAgent()
        
        score, fitness = trainer._evaluate_agent(agent)
        
        assert isinstance(score, int)
        assert isinstance(fitness, float)
        assert score >= 0
        assert fitness >= 0
        
    def test_neat_trainer_save_best_agent(self):
        """Test saving best agent."""
        trainer = NEATTrainer(max_generations=1, render_training=False)
        trainer.best_agent = NEATAgent()
        trainer.best_score = 10
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            original_models_dir = "models"
            if os.path.exists(original_models_dir):
                shutil.rmtree(original_models_dir)
                
            trainer._save_best_agent("test")
            
            # Check that file was created
            assert os.path.exists("models")
            assert len(os.listdir("models")) > 0
            
    def test_neat_trainer_get_training_stats(self):
        """Test training statistics retrieval."""
        trainer = NEATTrainer(max_generations=1, render_training=False)
        
        # Add some mock data
        trainer.best_fitness_history = [10.0, 15.0, 20.0]
        trainer.avg_fitness_history = [8.0, 12.0, 18.0]
        trainer.best_score_history = [1, 2, 3]
        trainer.best_score = 3
        
        stats = trainer.get_training_stats()
        
        assert 'generations_completed' in stats
        assert 'best_fitness' in stats
        assert 'best_score' in stats
        assert 'final_avg_fitness' in stats
        assert 'fitness_history' in stats
        assert 'avg_fitness_history' in stats
        assert 'score_history' in stats
        
        assert stats['generations_completed'] == 3
        assert stats['best_fitness'] == 20.0
        assert stats['best_score'] == 3

class TestDQNTrainer:
    """Test cases for the DQN trainer."""
    
    def test_dqn_trainer_initialization(self):
        """Test DQN trainer initialization."""
        trainer = DQNTrainer(max_episodes=100, render_training=False)
        
        assert trainer.max_episodes == 100
        assert trainer.max_fitness == 1000
        assert trainer.render_training == False
        assert trainer.save_interval == 100
        assert trainer.game is not None
        assert trainer.agent is not None
        assert trainer.best_score == 0
        assert trainer.best_episode == 0
        
    def test_dqn_trainer_save_agent(self):
        """Test saving agent."""
        trainer = DQNTrainer(max_episodes=1, render_training=False)
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            original_models_dir = "models"
            if os.path.exists(original_models_dir):
                shutil.rmtree(original_models_dir)
                
            trainer._save_agent("test")
            
            # Check that file was created
            assert os.path.exists("models")
            assert len(os.listdir("models")) > 0
            
    def test_dqn_trainer_get_training_stats(self):
        """Test training statistics retrieval."""
        trainer = DQNTrainer(max_episodes=1, render_training=False)
        
        # Add some mock data
        trainer.episode_rewards = [10.0, 15.0, 20.0]
        trainer.episode_scores = [1, 2, 3]
        trainer.epsilon_history = [1.0, 0.9, 0.8]
        trainer.avg_rewards = [12.0, 15.0, 18.0]
        trainer.avg_scores = [1.5, 2.0, 2.5]
        trainer.best_score = 3
        trainer.best_episode = 2
        
        stats = trainer.get_training_stats()
        
        assert 'episodes_completed' in stats
        assert 'best_score' in stats
        assert 'best_episode' in stats
        assert 'final_avg_reward' in stats
        assert 'final_avg_score' in stats
        assert 'final_epsilon' in stats
        assert 'reward_history' in stats
        assert 'score_history' in stats
        assert 'epsilon_history' in stats
        assert 'avg_reward_history' in stats
        assert 'avg_score_history' in stats
        
        assert stats['episodes_completed'] == 3
        assert stats['best_score'] == 3
        assert stats['best_episode'] == 2

class TestTrainingFunctions:
    """Test cases for the training convenience functions."""
    
    def test_train_neat_function(self):
        """Test the train_neat convenience function."""
        # Test with minimal parameters
        agent = train_neat(generations=1, population_size=5, render=False)
        
        assert isinstance(agent, NEATAgent)
        assert agent.name == "NEATAgent"
        
    def test_train_dqn_function(self):
        """Test the train_dqn convenience function."""
        # Test with minimal parameters
        agent = train_dqn(episodes=1, render=False)
        
        assert isinstance(agent, DQNAgent)
        assert agent.name == "DQNAgent"

class TestTrainingIntegration:
    """Integration tests for training."""
    
    def test_neat_training_integration(self):
        """Test complete NEAT training integration."""
        trainer = NEATTrainer(max_generations=2, render_training=False)
        
        # Run a short training session
        best_agent = trainer.train()
        
        assert best_agent is not None
        assert isinstance(best_agent, NEATAgent)
        assert len(trainer.best_fitness_history) > 0
        assert len(trainer.avg_fitness_history) > 0
        assert len(trainer.best_score_history) > 0
        
    def test_dqn_training_integration(self):
        """Test complete DQN training integration."""
        trainer = DQNTrainer(max_episodes=2, render_training=False)
        
        # Run a short training session
        agent = trainer.train()
        
        assert agent is not None
        assert isinstance(agent, DQNAgent)
        assert len(trainer.episode_rewards) > 0
        assert len(trainer.episode_scores) > 0
        assert len(trainer.epsilon_history) > 0
        
    def test_training_performance(self):
        """Test training performance (should be reasonable)."""
        import time
        
        # Test NEAT training performance
        start_time = time.time()
        trainer = NEATTrainer(max_generations=1, render_training=False)
        trainer.train()
        neat_time = time.time() - start_time
        
        # Test DQN training performance
        start_time = time.time()
        trainer = DQNTrainer(max_episodes=1, render_training=False)
        trainer.train()
        dqn_time = time.time() - start_time
        
        # Should complete in reasonable time
        assert neat_time < 60.0  # Less than 1 minute
        assert dqn_time < 60.0   # Less than 1 minute
        
    def test_training_consistency(self):
        """Test training consistency across multiple runs."""
        # Test NEAT consistency
        agents = []
        for _ in range(3):
            trainer = NEATTrainer(max_generations=1, render_training=False)
            agent = trainer.train()
            agents.append(agent)
            
        # All training runs should complete
        assert len(agents) == 3
        assert all(isinstance(agent, NEATAgent) for agent in agents)
        
        # Test DQN consistency
        agents = []
        for _ in range(3):
            trainer = DQNTrainer(max_episodes=1, render_training=False)
            agent = trainer.train()
            agents.append(agent)
            
        # All training runs should complete
        assert len(agents) == 3
        assert all(isinstance(agent, DQNAgent) for agent in agents)

class TestTrainingErrorHandling:
    """Test error handling in training."""
    
    def test_neat_trainer_invalid_parameters(self):
        """Test NEAT trainer with invalid parameters."""
        # Should handle invalid parameters gracefully
        trainer = NEATTrainer(max_generations=0, render_training=False)
        assert trainer.max_generations == 0
        
    def test_dqn_trainer_invalid_parameters(self):
        """Test DQN trainer with invalid parameters."""
        # Should handle invalid parameters gracefully
        trainer = DQNTrainer(max_episodes=0, render_training=False)
        assert trainer.max_episodes == 0
        
    def test_training_with_headless_game(self):
        """Test training with headless game mode."""
        # NEAT training
        trainer = NEATTrainer(max_generations=1, render_training=False)
        assert trainer.game.headless == True
        
        # DQN training
        trainer = DQNTrainer(max_episodes=1, render_training=False)
        assert trainer.game.headless == True

class TestTrainingStatistics:
    """Test training statistics and metrics."""
    
    def test_neat_training_statistics(self):
        """Test NEAT training statistics calculation."""
        trainer = NEATTrainer(max_generations=1, render_training=False)
        
        # Mock some training data
        trainer.best_fitness_history = [10.0, 15.0, 20.0]
        trainer.avg_fitness_history = [8.0, 12.0, 18.0]
        trainer.best_score_history = [1, 2, 3]
        
        stats = trainer.get_training_stats()
        
        # Check statistics
        assert stats['generations_completed'] == 3
        assert stats['best_fitness'] == 20.0
        assert stats['final_avg_fitness'] == 18.0
        assert len(stats['fitness_history']) == 3
        assert len(stats['avg_fitness_history']) == 3
        assert len(stats['score_history']) == 3
        
    def test_dqn_training_statistics(self):
        """Test DQN training statistics calculation."""
        trainer = DQNTrainer(max_episodes=1, render_training=False)
        
        # Mock some training data
        trainer.episode_rewards = [10.0, 15.0, 20.0]
        trainer.episode_scores = [1, 2, 3]
        trainer.epsilon_history = [1.0, 0.9, 0.8]
        trainer.avg_rewards = [12.0, 15.0, 18.0]
        trainer.avg_scores = [1.5, 2.0, 2.5]
        
        stats = trainer.get_training_stats()
        
        # Check statistics
        assert stats['episodes_completed'] == 3
        assert stats['final_avg_reward'] == 18.0
        assert stats['final_avg_score'] == 2.5
        assert stats['final_epsilon'] == 0.8
        assert len(stats['reward_history']) == 3
        assert len(stats['score_history']) == 3
        assert len(stats['epsilon_history']) == 3

if __name__ == "__main__":
    pytest.main([__file__]) 