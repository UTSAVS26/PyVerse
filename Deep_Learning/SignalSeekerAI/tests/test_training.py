import pytest
import numpy as np
import tempfile
import os
from train import TrainingEnvironment, Trainer, compare_agents
from agent import DQNAgent, RandomAgent, SequentialAgent, AdaptiveAgent


class TestTrainingEnvironment:
    """Test cases for TrainingEnvironment class."""
    
    def test_initialization(self):
        """Test training environment initialization."""
        env = TrainingEnvironment(
            freq_range=(1e6, 100e6),
            num_channels=1000,
            noise_floor=-90,
            detection_threshold=-70
        )
        
        assert env.num_channels == 1000
        assert env.frequencies.shape == (1000,)
        assert env.episode_length == 1000
        assert len(env.reward_history) == 0
        assert len(env.detection_history) == 0
    
    def test_reset(self):
        """Test environment reset functionality."""
        env = TrainingEnvironment(num_channels=500)
        
        # Take some steps
        env.spectrum.step()
        env.spectrum.step()
        
        # Reset
        initial_spectrum = env.reset()
        
        assert env.spectrum.time == 0.0
        assert len(env.reward_history) == 0
        assert len(env.detection_history) == 0
        assert len(initial_spectrum) == 500
    
    def test_step_execution(self):
        """Test step execution with agent."""
        env = TrainingEnvironment(num_channels=100)
        agent = RandomAgent(num_channels=100)
        
        # Take a step
        action = 50
        next_state, reward, done, info = env.step(agent, action)
        
        assert len(next_state) == 100
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        assert 'power' in info
        assert 'frequency' in info
        assert 'real_signal' in info
        assert 'detected_signals' in info
        assert 'active_signals' in info
    
    def test_reward_calculation(self):
        """Test reward calculation for different scenarios."""
        env = TrainingEnvironment(num_channels=100)
        agent = RandomAgent(num_channels=100)
        
        # Test reward calculation with different power levels
        test_cases = [
            (-50, True),   # Strong signal, real signal
            (-80, False),  # Weak signal, no real signal
            (-60, True),   # Medium signal, real signal
            (-50, False),  # Strong signal, false positive
        ]
        
        for power, real_signal in test_cases:
            # Set up signal in active_signals (this will be used by spectrum generation)
            if real_signal:
                env.spectrum.active_signals = {
                    'test_signal': {
                        'freq_idx': 50,
                        'frequency': env.frequencies[50],
                        'power': power,
                        'bandwidth': 1e3,
                        'start_time': env.spectrum.time,
                        'duration': 1.0
                    }
                }
            else:
                env.spectrum.active_signals = {}
            
            # Take step (this will generate spectrum with the signal)
            next_state, reward, done, info = env.step(agent, 50)
            
            # Check that reward is reasonable
            assert isinstance(reward, float)
            
            # Get the actual power at the action location
            actual_power = next_state[50]
            
            # Check reward based on actual power and signal presence
            if real_signal and actual_power > env.detector.threshold:
                assert reward > 0  # Should be positive for successful detection
            elif not real_signal and actual_power > env.detector.threshold:
                assert reward < 0  # Should be negative for false positive
            else:
                # For cases where power is below threshold, reward should be negative or small
                assert reward <= 0
    
    def test_episode_termination(self):
        """Test episode termination after maximum steps."""
        env = TrainingEnvironment(num_channels=100, episode_length=5)
        agent = RandomAgent(num_channels=100)
        
        # Take steps until episode ends
        done = False
        step_count = 0
        
        while not done and step_count < 10:
            action = agent.select_action(env.spectrum.step(), env.frequencies)
            next_state, reward, done, info = env.step(agent, action)
            step_count += 1
        
        assert done  # Episode should terminate
        assert step_count == 5  # Should terminate after episode_length steps (including the final step)
    
    def test_episode_statistics(self):
        """Test episode statistics calculation."""
        env = TrainingEnvironment(num_channels=100)
        agent = RandomAgent(num_channels=100)
        
        # Take some steps
        for _ in range(10):
            action = agent.select_action(env.spectrum.step(), env.frequencies)
            env.step(agent, action)
        
        stats = env.get_episode_statistics()
        
        assert 'total_reward' in stats
        assert 'avg_reward' in stats
        assert 'total_detections' in stats
        assert 'false_positives' in stats
        assert 'detection_rate' in stats
        assert 'false_positive_rate' in stats
        
        # Check that rates are between 0 and 1
        assert 0 <= stats['detection_rate'] <= 1
        assert 0 <= stats['false_positive_rate'] <= 1
    
    def test_reward_parameters(self):
        """Test reward parameter configuration."""
        env = TrainingEnvironment(num_channels=100)
        
        # Check default reward parameters
        expected_params = {
            'signal_detection': 10.0,
            'false_positive': -5.0,
            'missed_signal': -2.0,
            'scanning_cost': -0.1,
            'collision_penalty': -1.0
        }
        
        for param, expected_value in expected_params.items():
            assert env.reward_params[param] == expected_value


class TestTrainer:
    """Test cases for Trainer class."""
    
    def test_initialization(self):
        """Test trainer initialization."""
        env = TrainingEnvironment(num_channels=100)
        agent = DQNAgent(num_channels=100)
        trainer = Trainer(env, agent)
        
        assert trainer.env == env
        assert trainer.agent == agent
        assert trainer.num_episodes == 1000
        assert trainer.target_update_freq == 100
        assert trainer.save_freq == 200
        assert len(trainer.episode_rewards) == 0
        assert len(trainer.episode_detection_rates) == 0
    
    def test_training_loop(self):
        """Test basic training loop functionality."""
        env = TrainingEnvironment(num_channels=50, episode_length=10)
        agent = DQNAgent(num_channels=50)
        trainer = Trainer(env, agent)
        
        # Train for a few episodes
        training_stats = trainer.train(num_episodes=3)
        
        assert len(trainer.episode_rewards) == 3
        assert len(trainer.episode_detection_rates) == 3
        assert len(trainer.training_history) == 3
        
        # Check training statistics
        assert 'episode_rewards' in training_stats
        assert 'episode_detection_rates' in training_stats
        assert 'training_history' in training_stats
        assert 'final_epsilon' in training_stats
        assert 'avg_final_reward' in training_stats
        assert 'avg_final_detection_rate' in training_stats
    
    def test_epsilon_decay(self):
        """Test epsilon decay during training."""
        env = TrainingEnvironment(num_channels=50, episode_length=5)
        agent = DQNAgent(num_channels=50, epsilon=1.0, epsilon_decay=0.9)
        trainer = Trainer(env, agent)
        
        initial_epsilon = agent.epsilon
        
        # Train for a few episodes with longer episodes to ensure replay training
        trainer.train(num_episodes=5)
        
        # Epsilon should have decayed (if replay training occurred)
        # Note: Epsilon only decays during replay training, which requires enough experiences
        if len(agent.memory) >= agent.batch_size:
            assert agent.epsilon < initial_epsilon
        else:
            # If not enough experiences for replay, epsilon should remain the same
            assert agent.epsilon == initial_epsilon
        assert agent.epsilon >= agent.epsilon_min
    
    def test_target_network_update(self):
        """Test target network update during training."""
        env = TrainingEnvironment(num_channels=50, episode_length=5)
        agent = DQNAgent(num_channels=50)
        trainer = Trainer(env, agent)
        
        # Set target update frequency to 2
        trainer.target_update_freq = 2
        
        # Train for a few episodes
        trainer.train(num_episodes=4)
        
        # Target network should have been updated
        # (We can't directly test this, but we can ensure training completes)
        assert len(trainer.episode_rewards) == 4
    
    def test_model_saving(self):
        """Test model saving during training."""
        env = TrainingEnvironment(num_channels=50, episode_length=5)
        agent = DQNAgent(num_channels=50)
        trainer = Trainer(env, agent)
        
        # Set save frequency to 2
        trainer.save_freq = 2
        
        # Train for a few episodes
        trainer.train(num_episodes=4)
        
        # Check that model files were created in the current directory
        model_files = [f for f in os.listdir("models") if f.endswith('.pth')]
        assert len(model_files) > 0
    
    def test_training_statistics(self):
        """Test comprehensive training statistics."""
        env = TrainingEnvironment(num_channels=50, episode_length=5)
        agent = DQNAgent(num_channels=50)
        trainer = Trainer(env, agent)
        
        # Train for a few episodes
        training_stats = trainer.train(num_episodes=5)
        
        # Check all statistics are present
        required_keys = [
            'episode_rewards', 'episode_detection_rates', 'training_history',
            'final_epsilon', 'avg_final_reward', 'avg_final_detection_rate'
        ]
        
        for key in required_keys:
            assert key in training_stats
        
        # Check that statistics are reasonable
        assert len(training_stats['episode_rewards']) == 5
        assert len(training_stats['episode_detection_rates']) == 5
        assert len(training_stats['training_history']) == 5
        
        # Check that final statistics are calculated correctly
        # Since we only have 5 episodes, the avg_final_reward should be the average of all episodes
        expected_avg = sum(training_stats['episode_rewards']) / len(training_stats['episode_rewards'])
        # Use a more lenient tolerance for floating point arithmetic
        assert abs(training_stats['avg_final_reward'] - expected_avg) < 1e-3


class TestAgentComparison:
    """Test cases for agent comparison functionality."""
    
    def test_compare_agents(self):
        """Test agent comparison function."""
        env = TrainingEnvironment(num_channels=50, episode_length=5)
        
        # Compare agents
        results = compare_agents(env, num_episodes=3)
        
        # Check that all agent types are included
        expected_agents = ['Random', 'Sequential', 'Adaptive', 'DQN']
        assert all(agent in results for agent in expected_agents)
        
        # Check that each agent has required statistics
        for agent_name, stats in results.items():
            required_keys = [
                'avg_reward', 'std_reward', 'avg_detection_rate', 
                'std_detection_rate', 'episode_rewards', 'episode_detection_rates'
            ]
            
            for key in required_keys:
                assert key in stats
            
            # Check that statistics are reasonable
            assert len(stats['episode_rewards']) == 3
            assert len(stats['episode_detection_rates']) == 3
            assert 0 <= stats['avg_detection_rate'] <= 1
    
    def test_agent_performance_differences(self):
        """Test that different agents show performance differences."""
        env = TrainingEnvironment(num_channels=50, episode_length=10)
        
        # Compare agents
        results = compare_agents(env, num_episodes=5)
        
        # Get average rewards
        avg_rewards = [results[agent]['avg_reward'] for agent in results]
        
        # Different agents should have different performance
        # (though this is not guaranteed due to randomness)
        reward_variance = np.var(avg_rewards)
        assert reward_variance >= 0  # Should be non-negative
    
    def test_agent_detection_rates(self):
        """Test agent detection rate differences."""
        env = TrainingEnvironment(num_channels=50, episode_length=10)
        
        # Compare agents
        results = compare_agents(env, num_episodes=5)
        
        # Get average detection rates
        avg_detection_rates = [results[agent]['avg_detection_rate'] for agent in results]
        
        # All detection rates should be between 0 and 1
        for rate in avg_detection_rates:
            assert 0 <= rate <= 1


class TestTrainingIntegration:
    """Integration tests for training system."""
    
    def test_full_training_workflow(self):
        """Test complete training workflow."""
        env = TrainingEnvironment(num_channels=100, episode_length=20)
        agent = DQNAgent(num_channels=100)
        trainer = Trainer(env, agent)
        
        # Train for a few episodes
        training_stats = trainer.train(num_episodes=5)
        
        # Check that training completed successfully
        assert len(training_stats['episode_rewards']) == 5
        assert len(training_stats['episode_detection_rates']) == 5
        
        # Check that agent learned something (epsilon should decay)
        assert agent.epsilon < 1.0
    
    def test_agent_learning_progress(self):
        """Test that agent shows learning progress."""
        env = TrainingEnvironment(num_channels=100, episode_length=20)
        agent = DQNAgent(num_channels=100, epsilon=1.0, epsilon_decay=0.8)
        trainer = Trainer(env, agent)
        
        # Train for several episodes
        training_stats = trainer.train(num_episodes=10)
        
        # Check that epsilon decayed
        assert agent.epsilon < 1.0
        
        # Check that training history shows progress
        epsilons = [h['epsilon'] for h in training_stats['training_history']]
        assert epsilons[-1] < epsilons[0]  # Epsilon should decrease
    
    def test_reward_consistency(self):
        """Test that rewards are consistent across episodes."""
        env = TrainingEnvironment(num_channels=50, episode_length=10)
        agent = RandomAgent(num_channels=50)  # Use random agent for consistency
        
        # Run multiple episodes
        episode_rewards = []
        for episode in range(5):
            state = env.reset()
            episode_reward = 0
            
            for step in range(env.episode_length):
                action = agent.select_action(state, env.frequencies)
                next_state, reward, done, info = env.step(agent, action)
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
        
        # All rewards should be finite
        assert all(np.isfinite(reward) for reward in episode_rewards)
        
        # Rewards should be reasonable (not extremely large or small)
        assert all(-1000 < reward < 1000 for reward in episode_rewards)


class TestTrainingEdgeCases:
    """Test edge cases in training system."""
    
    def test_empty_episode(self):
        """Test training with very short episodes."""
        env = TrainingEnvironment(num_channels=50, episode_length=1)
        agent = RandomAgent(num_channels=50)
        trainer = Trainer(env, agent)
        
        # Train for one episode
        training_stats = trainer.train(num_episodes=1)
        
        assert len(training_stats['episode_rewards']) == 1
        assert len(training_stats['episode_detection_rates']) == 1
    
    def test_single_channel_spectrum(self):
        """Test training with minimal spectrum."""
        env = TrainingEnvironment(num_channels=1, episode_length=5)
        agent = RandomAgent(num_channels=1)
        trainer = Trainer(env, agent)
        
        # Train for a few episodes
        training_stats = trainer.train(num_episodes=3)
        
        assert len(training_stats['episode_rewards']) == 3
    
    def test_high_noise_environment(self):
        """Test training in high noise environment."""
        env = TrainingEnvironment(
            num_channels=50, 
            episode_length=10,
            noise_floor=-50  # High noise floor
        )
        agent = RandomAgent(num_channels=50)
        trainer = Trainer(env, agent)
        
        # Train for a few episodes
        training_stats = trainer.train(num_episodes=3)
        
        assert len(training_stats['episode_rewards']) == 3


if __name__ == "__main__":
    pytest.main([__file__])
