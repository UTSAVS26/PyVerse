import pytest
import numpy as np
import tempfile
import os
import sys
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.visualization import TrainingVisualizer
from utils.logger import TrainingLogger

class TestTrainingVisualizer:
    """Test cases for TrainingVisualizer"""
    
    def setup_method(self):
        """Setup method called before each test"""
        self.temp_dir = tempfile.mkdtemp()
        self.visualizer = TrainingVisualizer(save_dir=self.temp_dir)
    
    def teardown_method(self):
        """Teardown method called after each test"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test visualizer initialization"""
        assert self.visualizer.save_dir == self.temp_dir
        assert os.path.exists(self.temp_dir)
    
    @patch('matplotlib.pyplot.show')
    def test_plot_training_progress(self, mock_show):
        """Test training progress plotting"""
        episodes = list(range(1, 101))
        scores = np.random.randint(0, 20, 100).tolist()
        rewards = np.random.uniform(-10, 10, 100).tolist()
        
        # Should not raise an exception
        self.visualizer.plot_training_progress(episodes, scores, rewards, "TestAgent")
        
        # Check if plot was saved
        plot_files = [f for f in os.listdir(self.temp_dir) if f.endswith('.png')]
        assert len(plot_files) > 0
    
    @patch('matplotlib.pyplot.show')
    def test_plot_average_scores(self, mock_show):
        """Test average scores plotting"""
        episodes = list(range(1, 101))
        avg_scores = np.random.uniform(5, 15, 100).tolist()
        
        # Should not raise an exception
        self.visualizer.plot_average_scores(episodes, avg_scores, "TestAgent")
        
        # Check if plot was saved
        plot_files = [f for f in os.listdir(self.temp_dir) if f.endswith('.png')]
        assert len(plot_files) > 0
    
    @patch('matplotlib.pyplot.show')
    def test_plot_epsilon_decay(self, mock_show):
        """Test epsilon decay plotting"""
        episodes = list(range(1, 101))
        epsilons = [0.995 ** i for i in range(100)]
        
        # Should not raise an exception
        self.visualizer.plot_epsilon_decay(episodes, epsilons, "TestAgent")
        
        # Check if plot was saved
        plot_files = [f for f in os.listdir(self.temp_dir) if f.endswith('.png')]
        assert len(plot_files) > 0
    
    @patch('matplotlib.pyplot.show')
    def test_plot_comparison(self, mock_show):
        """Test agent comparison plotting"""
        results = {
            'Agent1': {
                'episodes': list(range(1, 51)),
                'scores': np.random.randint(0, 20, 50).tolist(),
                'rewards': np.random.uniform(-10, 10, 50).tolist()
            },
            'Agent2': {
                'episodes': list(range(1, 51)),
                'scores': np.random.randint(0, 20, 50).tolist(),
                'rewards': np.random.uniform(-10, 10, 50).tolist()
            }
        }
        
        # Should not raise an exception
        self.visualizer.plot_comparison(results)
        
        # Check if plot was saved
        plot_files = [f for f in os.listdir(self.temp_dir) if f.endswith('.png')]
        assert len(plot_files) > 0
    
    @patch('matplotlib.pyplot.show')
    def test_plot_histogram(self, mock_show):
        """Test score histogram plotting"""
        scores = np.random.randint(0, 20, 100).tolist()
        
        # Should not raise an exception
        self.visualizer.plot_histogram(scores, "TestAgent")
        
        # Check if plot was saved
        plot_files = [f for f in os.listdir(self.temp_dir) if f.endswith('.png')]
        assert len(plot_files) > 0
    
    @patch('matplotlib.pyplot.show')
    def test_create_summary_plot(self, mock_show):
        """Test summary plot creation"""
        training_data = {
            'episodes': list(range(1, 101)),
            'scores': np.random.randint(0, 20, 100).tolist(),
            'rewards': np.random.uniform(-10, 10, 100).tolist(),
            'avg_scores': np.random.uniform(5, 15, 100).tolist(),
            'epsilons': [0.995 ** i for i in range(100)]
        }
        
        # Should not raise an exception
        self.visualizer.create_summary_plot(training_data, "TestAgent")
        
        # Check if plot was saved
        plot_files = [f for f in os.listdir(self.temp_dir) if f.endswith('.png')]
        assert len(plot_files) > 0
    
    @patch('matplotlib.pyplot.show')
    def test_create_summary_plot_missing_data(self, mock_show):
        """Test summary plot with missing optional data"""
        training_data = {
            'episodes': list(range(1, 101)),
            'scores': np.random.randint(0, 20, 100).tolist(),
            'rewards': np.random.uniform(-10, 10, 100).tolist()
            # Missing avg_scores and epsilons
        }
        
        # Should not raise an exception
        self.visualizer.create_summary_plot(training_data, "TestAgent")
        
        # Check if plot was saved
        plot_files = [f for f in os.listdir(self.temp_dir) if f.endswith('.png')]
        assert len(plot_files) > 0

class TestTrainingLogger:
    """Test cases for TrainingLogger"""
    
    def setup_method(self):
        """Setup method called before each test"""
        self.temp_dir = tempfile.mkdtemp()
        self.logger = TrainingLogger(log_dir=self.temp_dir, agent_name="TestAgent")
    
    def teardown_method(self):
        """Teardown method called after each test"""
        self.logger.close()
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test logger initialization"""
        assert self.logger.log_dir == self.temp_dir
        assert self.logger.agent_name == "TestAgent"
        assert os.path.exists(self.temp_dir)
        assert len(self.logger.episodes) == 0
        assert len(self.logger.scores) == 0
        assert len(self.logger.rewards) == 0
        assert self.logger.start_time is None
    
    def test_start_training(self):
        """Test training start logging"""
        agent_info = {'type': 'TestAgent', 'epsilon': 1.0}
        
        self.logger.start_training(1000, agent_info)
        
        assert self.logger.start_time is not None
    
    def test_log_episode(self):
        """Test episode logging"""
        episode = 1
        score = 5.0
        reward = 10.5
        epsilon = 0.9
        avg_score = 4.5
        
        self.logger.log_episode(episode, score, reward, epsilon, avg_score)
        
        assert len(self.logger.episodes) == 1
        assert self.logger.episodes[0] == episode
        assert self.logger.scores[0] == score
        assert self.logger.rewards[0] == reward
        assert self.logger.epsilons[0] == epsilon
        assert len(self.logger.avg_scores) == 1
        assert self.logger.avg_scores[0] == avg_score
    
    def test_log_episode_without_avg_score(self):
        """Test episode logging without average score"""
        episode = 1
        score = 5.0
        reward = 10.5
        epsilon = 0.9
        
        self.logger.log_episode(episode, score, reward, epsilon)
        
        assert len(self.logger.episodes) == 1
        assert len(self.logger.avg_scores) == 0
    
    def test_log_training_complete(self):
        """Test training completion logging"""
        # Start training first
        self.logger.start_training(100, {})
        
        best_score = 15.0
        final_avg_score = 12.5
        
        self.logger.log_training_complete(best_score, final_avg_score)
        
        # Should not raise an exception
        assert True
    
    def test_get_training_data(self):
        """Test training data retrieval"""
        # Add some data
        self.logger.episodes = [1, 2, 3]
        self.logger.scores = [5, 6, 7]
        self.logger.rewards = [10, 11, 12]
        self.logger.epsilons = [0.9, 0.8, 0.7]
        self.logger.avg_scores = [5.5, 6.5, 7.5]
        
        data = self.logger.get_training_data()
        
        assert data['episodes'] == [1, 2, 3]
        assert data['scores'] == [5, 6, 7]
        assert data['rewards'] == [10, 11, 12]
        assert data['epsilons'] == [0.9, 0.8, 0.7]
        assert data['avg_scores'] == [5.5, 6.5, 7.5]
    
    def test_get_training_data_without_avg_scores(self):
        """Test training data retrieval without average scores"""
        # Add some data without avg_scores
        self.logger.episodes = [1, 2, 3]
        self.logger.scores = [5, 6, 7]
        self.logger.rewards = [10, 11, 12]
        self.logger.epsilons = [0.9, 0.8, 0.7]
        
        data = self.logger.get_training_data()
        
        assert data['episodes'] == [1, 2, 3]
        assert data['scores'] == [5, 6, 7]
        assert data['rewards'] == [10, 11, 12]
        assert data['epsilons'] == [0.9, 0.8, 0.7]
        assert 'avg_scores' not in data
    
    def test_log_agent_info(self):
        """Test agent info logging"""
        agent_info = {'type': 'TestAgent', 'epsilon': 1.0, 'learning_rate': 0.1}
        
        # Should not raise an exception
        self.logger.log_agent_info(agent_info)
        assert True
    
    def test_log_error(self):
        """Test error logging"""
        error_msg = "Test error message"
        
        # Should not raise an exception
        self.logger.log_error(error_msg)
        assert True
    
    def test_log_warning(self):
        """Test warning logging"""
        warning_msg = "Test warning message"
        
        # Should not raise an exception
        self.logger.log_warning(warning_msg)
        assert True
    
    def test_log_hyperparameters(self):
        """Test hyperparameters logging"""
        hyperparams = {'learning_rate': 0.1, 'epsilon': 1.0, 'batch_size': 32}
        
        # Should not raise an exception
        self.logger.log_hyperparameters(hyperparams)
        assert True
    
    def test_log_performance_metrics(self):
        """Test performance metrics logging"""
        metrics = {'accuracy': 0.95, 'loss': 0.05, 'reward': 10.5}
        
        # Should not raise an exception
        self.logger.log_performance_metrics(metrics)
        assert True
    
    def test_log_file_creation(self):
        """Test that log file is created"""
        # Start training and log some episodes
        self.logger.start_training(100, {})
        self.logger.log_episode(1, 5.0, 10.0, 0.9)
        self.logger.log_episode(2, 6.0, 11.0, 0.8)
        
        # Check if log file exists
        log_files = [f for f in os.listdir(self.temp_dir) if f.endswith('.log')]
        assert len(log_files) > 0
        
        # Check log file content
        log_file_path = os.path.join(self.temp_dir, log_files[0])
        with open(log_file_path, 'r') as f:
            content = f.read()
            assert "TestAgent" in content
            assert "Episode 1" in content
            assert "Episode 2" in content
    
    def test_multiple_episodes_logging(self):
        """Test logging multiple episodes"""
        self.logger.start_training(100, {})
        
        for i in range(10):
            self.logger.log_episode(i+1, float(i), float(i*2), 0.9 - i*0.01)
        
        assert len(self.logger.episodes) == 10
        assert len(self.logger.scores) == 10
        assert len(self.logger.rewards) == 10
        assert len(self.logger.epsilons) == 10
        
        # Check values
        assert self.logger.episodes == list(range(1, 11))
        assert self.logger.scores == list(range(10))
        assert self.logger.rewards == [i*2 for i in range(10)]
        assert self.logger.epsilons == [0.9 - i*0.01 for i in range(10)]
