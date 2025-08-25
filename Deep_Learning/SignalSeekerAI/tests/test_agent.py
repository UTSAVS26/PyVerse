import pytest
import numpy as np
from agent import (
    BaseAgent, RandomAgent, SequentialAgent, AdaptiveAgent, 
    DQNAgent, DQNetwork, MultiAgentEnvironment
)


class TestBaseAgent:
    """Test cases for BaseAgent abstract class."""
    
    def test_base_agent_initialization(self):
        """Test base agent initialization."""
        # Create a concrete implementation for testing
        class TestAgent(BaseAgent):
            def select_action(self, spectrum, frequencies):
                return 0
        
        agent = TestAgent(num_channels=100, scan_window=10)
        
        assert agent.num_channels == 100
        assert agent.scan_window == 10
        assert agent.current_position == 0
        assert len(agent.scan_history) == 0
        assert len(agent.detection_history) == 0
    
    def test_scan_spectrum(self):
        """Test spectrum scanning functionality."""
        class TestAgent(BaseAgent):
            def select_action(self, spectrum, frequencies):
                return 50  # Always scan at position 50
        
        agent = TestAgent(num_channels=100)
        spectrum = np.random.uniform(-90, -30, 100)
        frequencies = np.linspace(1e6, 100e6, 100)
        
        scan_result = agent.scan_spectrum(spectrum, frequencies)
        
        assert scan_result['position'] == 50
        assert scan_result['frequency'] == frequencies[50]
        assert scan_result['power'] == spectrum[50]
        assert scan_result['timestamp'] == 0
        assert len(agent.scan_history) == 1
    
    def test_scan_statistics(self):
        """Test scan statistics calculation."""
        class TestAgent(BaseAgent):
            def select_action(self, spectrum, frequencies):
                return np.random.randint(0, len(spectrum))
        
        agent = TestAgent(num_channels=100)
        spectrum = np.random.uniform(-90, -30, 100)
        frequencies = np.linspace(1e6, 100e6, 100)
        
        # Perform several scans
        for _ in range(10):
            agent.scan_spectrum(spectrum, frequencies)
        
        stats = agent.get_scan_statistics()
        
        assert stats['total_scans'] == 10
        assert 'avg_power' in stats
        assert 'max_power' in stats
        assert 'min_power' in stats
        assert 'position_variance' in stats
        assert 'coverage' in stats
        assert 0 <= stats['coverage'] <= 1


class TestRandomAgent:
    """Test cases for RandomAgent class."""
    
    def test_initialization(self):
        """Test random agent initialization."""
        agent = RandomAgent(num_channels=100)
        assert agent.num_channels == 100
        assert isinstance(agent, BaseAgent)
    
    def test_action_selection(self):
        """Test random action selection."""
        agent = RandomAgent(num_channels=100)
        spectrum = np.random.uniform(-90, -30, 100)
        frequencies = np.linspace(1e6, 100e6, 100)
        
        # Test multiple actions
        actions = []
        for _ in range(100):
            action = agent.select_action(spectrum, frequencies)
            actions.append(action)
        
        # Check that actions are within valid range
        assert all(0 <= action < 100 for action in actions)
        
        # Check that actions are random (not all the same)
        assert len(set(actions)) > 1
    
    def test_action_distribution(self):
        """Test that actions are reasonably distributed."""
        agent = RandomAgent(num_channels=10)
        spectrum = np.random.uniform(-90, -30, 10)
        frequencies = np.linspace(1e6, 10e6, 10)
        
        # Collect many actions
        actions = []
        for _ in range(1000):
            action = agent.select_action(spectrum, frequencies)
            actions.append(action)
        
        # Check that all positions are visited
        unique_actions = set(actions)
        assert len(unique_actions) == 10  # All positions visited


class TestSequentialAgent:
    """Test cases for SequentialAgent class."""
    
    def test_initialization(self):
        """Test sequential agent initialization."""
        agent = SequentialAgent(num_channels=100)
        assert agent.num_channels == 100
        assert agent.current_position == 0
    
    def test_sequential_action_selection(self):
        """Test sequential action selection."""
        agent = SequentialAgent(num_channels=5)
        spectrum = np.random.uniform(-90, -30, 5)
        frequencies = np.linspace(1e6, 5e6, 5)
        
        # Test sequential progression
        expected_sequence = [1, 2, 3, 4, 0, 1, 2, 3, 4, 0]  # Wraps around, starts from current_position=0
        
        for i, expected in enumerate(expected_sequence[:10]):
            action = agent.select_action(spectrum, frequencies)
            assert action == expected
            agent.current_position = action  # Update position manually
    
    def test_wraparound_behavior(self):
        """Test that sequential agent wraps around at boundaries."""
        agent = SequentialAgent(num_channels=3)
        spectrum = np.random.uniform(-90, -30, 3)
        frequencies = np.linspace(1e6, 3e6, 3)
        
        # Set position to last element
        agent.current_position = 2
        
        # Next action should wrap to 0
        action = agent.select_action(spectrum, frequencies)
        assert action == 0


class TestAdaptiveAgent:
    """Test cases for AdaptiveAgent class."""
    
    def test_initialization(self):
        """Test adaptive agent initialization."""
        agent = AdaptiveAgent(num_channels=100, power_threshold=-70, exploration_rate=0.1)
        assert agent.num_channels == 100
        assert agent.power_threshold == -70
        assert agent.exploration_rate == 0.1
    
    def test_exploration_behavior(self):
        """Test exploration behavior with high exploration rate."""
        agent = AdaptiveAgent(num_channels=10, exploration_rate=1.0)  # Always explore
        spectrum = np.random.uniform(-90, -30, 10)
        frequencies = np.linspace(1e6, 10e6, 10)
        
        # Should always choose random actions
        actions = []
        for _ in range(50):
            action = agent.select_action(spectrum, frequencies)
            actions.append(action)
        
        # Should have variety in actions
        assert len(set(actions)) > 1
    
    def test_exploitation_behavior(self):
        """Test exploitation behavior with low exploration rate."""
        agent = AdaptiveAgent(num_channels=10, exploration_rate=0.0)  # Never explore
        spectrum = np.full(10, -90)  # All below threshold
        spectrum[5] = -50  # One high-power signal
        frequencies = np.linspace(1e6, 10e6, 10)
        
        # Should always choose the high-power position
        for _ in range(10):
            action = agent.select_action(spectrum, frequencies)
            assert action == 5
    
    def test_mixed_behavior(self):
        """Test mixed exploration/exploitation behavior."""
        agent = AdaptiveAgent(num_channels=10, exploration_rate=0.5)
        spectrum = np.full(10, -90)
        spectrum[3] = -50  # High-power signal
        frequencies = np.linspace(1e6, 10e6, 10)
        
        # Collect actions
        actions = []
        for _ in range(100):
            action = agent.select_action(spectrum, frequencies)
            actions.append(action)
        
        # Should have some preference for high-power position
        high_power_count = actions.count(3)
        assert high_power_count > 10  # Should prefer high-power position


class TestDQNAgent:
    """Test cases for DQNAgent class."""
    
    def test_initialization(self):
        """Test DQN agent initialization."""
        agent = DQNAgent(num_channels=100)
        assert agent.num_channels == 100
        assert agent.epsilon == 1.0
        assert agent.epsilon_min == 0.01
        assert agent.epsilon_decay == 0.995
        assert len(agent.memory) == 0
    
    def test_action_selection_exploration(self):
        """Test action selection during exploration phase."""
        agent = DQNAgent(num_channels=10, epsilon=1.0)  # Always explore
        spectrum = np.random.uniform(-90, -30, 10)
        frequencies = np.linspace(1e6, 10e6, 10)
        
        actions = []
        for _ in range(50):
            action = agent.select_action(spectrum, frequencies)
            actions.append(action)
        
        # Should have variety in actions during exploration
        assert len(set(actions)) > 1
    
    def test_state_representation(self):
        """Test state representation generation."""
        agent = DQNAgent(num_channels=5)
        spectrum = np.array([-90, -80, -70, -60, -50])
        
        state = agent._get_state_representation(spectrum)
        
        # State should be normalized spectrum + position one-hot
        expected_length = 5 + 5  # spectrum + position one-hot
        assert len(state) == expected_length
        
        # Check normalization (should be between 0 and 1)
        spectrum_part = state[:5]
        assert np.all(spectrum_part >= 0)
        assert np.all(spectrum_part <= 1)
        
        # Check position one-hot
        position_part = state[5:]
        assert np.sum(position_part) == 1.0  # Only one position should be 1
    
    def test_memory_management(self):
        """Test experience replay memory."""
        agent = DQNAgent(num_channels=5, memory_size=5)
        
        # Add experiences
        for i in range(10):
            state = np.random.rand(10)
            action = i % 5
            reward = i
            next_state = np.random.rand(10)
            done = i == 9
            
            agent.remember(state, action, reward, next_state, done)
        
        # Memory should be limited to memory_size
        assert len(agent.memory) <= 5
    
    def test_replay_training(self):
        """Test replay training functionality."""
        agent = DQNAgent(num_channels=5)
        
        # Add enough experiences to trigger replay (need at least batch_size=32)
        for i in range(32):
            state = np.random.rand(10)
            action = i % 5
            reward = i
            next_state = np.random.rand(10)
            done = False
            
            agent.remember(state, action, reward, next_state, done)
        
        # Store initial epsilon
        initial_epsilon = agent.epsilon
        
        # Should be able to replay
        agent.replay()
        
        # Epsilon should decay
        assert agent.epsilon < initial_epsilon


class TestDQNetwork:
    """Test cases for DQNetwork class."""
    
    def test_initialization(self):
        """Test DQNetwork initialization."""
        network = DQNetwork(num_channels=10, scan_window=5)
        assert network is not None
    
    def test_forward_pass(self):
        """Test forward pass through the network."""
        network = DQNetwork(num_channels=5, scan_window=3)
        
        # Input size should be num_channels * 2
        input_size = 5 * 2
        x = np.random.rand(1, input_size)
        
        # Convert to tensor
        import torch
        x_tensor = torch.FloatTensor(x)
        
        # Forward pass
        output = network(x_tensor)
        
        # Output should have num_channels values
        assert output.shape == (1, 5)
    
    def test_network_architecture(self):
        """Test network architecture."""
        network = DQNetwork(num_channels=10, scan_window=5)
        
        # Check that network has the expected layers
        assert len(network.network) > 0
        
        # Check input and output sizes
        input_size = 10 * 2
        x = np.random.rand(1, input_size)
        
        import torch
        x_tensor = torch.FloatTensor(x)
        output = network(x_tensor)
        
        assert output.shape[1] == 10  # Output size should match num_channels


class TestMultiAgentEnvironment:
    """Test cases for MultiAgentEnvironment class."""
    
    def test_initialization(self):
        """Test multi-agent environment initialization."""
        env = MultiAgentEnvironment(num_agents=4, num_channels=100)
        assert env.num_agents == 4
        assert env.num_channels == 100
        assert len(env.agents) == 4
    
    def test_agent_creation(self):
        """Test that different agent types are created."""
        env = MultiAgentEnvironment(num_agents=4, num_channels=100)
        
        agent_types = [type(agent).__name__ for agent in env.agents]
        expected_types = ['RandomAgent', 'SequentialAgent', 'AdaptiveAgent', 'DQNAgent']
        
        assert agent_types == expected_types
    
    def test_step_execution(self):
        """Test step execution with multiple agents."""
        env = MultiAgentEnvironment(num_agents=3, num_channels=10)
        spectrum = np.random.uniform(-90, -30, 10)
        frequencies = np.linspace(1e6, 10e6, 10)
        
        results = env.step(spectrum, frequencies)
        
        assert len(results) == 3  # One result per agent
        for result in results:
            assert 'agent_id' in result
            assert 'collision' in result
            assert 'position' in result
            assert 'power' in result
    
    def test_collision_detection(self):
        """Test collision detection between agents."""
        env = MultiAgentEnvironment(num_agents=2, num_channels=10)
        
        # Mock actions that would cause collision
        actions = [5, 5]  # Both agents choose same position
        
        collisions = env._check_collisions(actions)
        
        assert collisions == [True, True]  # Both agents should have collision
    
    def test_no_collision_detection(self):
        """Test collision detection when no collision occurs."""
        env = MultiAgentEnvironment(num_agents=2, num_channels=10)
        
        # Mock actions that don't cause collision
        actions = [3, 7]  # Different positions
        
        collisions = env._check_collisions(actions)
        
        assert collisions == [False, False]  # No collisions
    
    def test_competition_statistics(self):
        """Test competition statistics generation."""
        env = MultiAgentEnvironment(num_agents=2, num_channels=10)
        spectrum = np.random.uniform(-90, -30, 10)
        frequencies = np.linspace(1e6, 10e6, 10)
        
        # Perform some steps
        for _ in range(5):
            env.step(spectrum, frequencies)
        
        stats = env.get_competition_statistics()
        
        assert len(stats) == 2  # One entry per agent
        for agent_id in ['agent_0', 'agent_1']:
            assert agent_id in stats
            agent_stats = stats[agent_id]
            assert 'agent_type' in agent_stats
            assert 'total_scans' in agent_stats


class TestAgentIntegration:
    """Integration tests for agent system."""
    
    def test_agent_spectrum_interaction(self):
        """Test agent interaction with spectrum."""
        from spectrum import RadioSpectrum
        
        spectrum = RadioSpectrum(num_channels=100)
        agent = RandomAgent(num_channels=100)
        
        # Get current spectrum
        current_spectrum = spectrum.step()
        
        # Agent scan
        scan_result = agent.scan_spectrum(current_spectrum, spectrum.frequencies)
        
        assert scan_result['position'] >= 0
        assert scan_result['position'] < 100
        assert scan_result['power'] == current_spectrum[scan_result['position']]
    
    def test_multiple_agent_types(self):
        """Test different agent types with same spectrum."""
        from spectrum import RadioSpectrum
        
        spectrum = RadioSpectrum(num_channels=50)
        current_spectrum = spectrum.step()
        frequencies = spectrum.frequencies
        
        agents = [
            RandomAgent(num_channels=50),
            SequentialAgent(num_channels=50),
            AdaptiveAgent(num_channels=50),
            DQNAgent(num_channels=50)
        ]
        
        results = []
        for agent in agents:
            scan_result = agent.scan_spectrum(current_spectrum, frequencies)
            results.append(scan_result)
        
        # All agents should produce valid results
        assert len(results) == 4
        for result in results:
            assert 'position' in result
            assert 'power' in result
            assert 'frequency' in result


if __name__ == "__main__":
    pytest.main([__file__])
