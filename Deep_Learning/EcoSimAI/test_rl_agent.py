import pytest
import numpy as np
import torch
from environment import Environment, Position, CellType
from agents import Prey, Predator
from rl_agent import QNetwork, RLAgent, QLearningAgent, DQNAgent, RLPrey, RLPredator

class TestQNetwork:
    """Test cases for the QNetwork neural network."""
    
    def test_qnetwork_initialization(self):
        """Test QNetwork initialization."""
        input_size = 10
        output_size = 5
        hidden_size = 32
        
        network = QNetwork(input_size, output_size, hidden_size)
        
        assert network.fc1.in_features == input_size
        assert network.fc1.out_features == hidden_size
        assert network.fc2.in_features == hidden_size
        assert network.fc2.out_features == hidden_size
        assert network.fc3.in_features == hidden_size
        assert network.fc3.out_features == output_size
    
    def test_qnetwork_forward_pass(self):
        """Test QNetwork forward pass."""
        input_size = 10
        output_size = 5
        network = QNetwork(input_size, output_size)
        
        # Create input tensor
        x = torch.randn(1, input_size)
        
        # Forward pass
        output = network(x)
        
        assert output.shape == (1, output_size)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_qnetwork_batch_forward_pass(self):
        """Test QNetwork forward pass with batch input."""
        input_size = 10
        output_size = 5
        batch_size = 4
        network = QNetwork(input_size, output_size)
        
        # Create batch input tensor
        x = torch.randn(batch_size, input_size)
        
        # Forward pass
        output = network(x)
        
        assert output.shape == (batch_size, output_size)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

class TestRLAgent:
    """Test cases for the base RLAgent class."""
    
    def test_rl_agent_initialization(self):
        """Test RLAgent initialization."""
        pos = Position(5, 5)
        agent = RLAgent(0, pos)
        
        assert agent.learning_rate == 0.001
        assert agent.epsilon == 0.1
        assert agent.gamma == 0.95
        assert len(agent.memory) == 0
        assert agent.batch_size == 32
        assert agent.update_frequency == 100
        assert agent.step_count == 0
    
    def test_rl_agent_custom_initialization(self):
        """Test RLAgent initialization with custom parameters."""
        pos = Position(5, 5)
        agent = RLAgent(0, pos, learning_rate=0.01, epsilon=0.2, gamma=0.9)
        
        assert agent.learning_rate == 0.01
        assert agent.epsilon == 0.2
        assert agent.gamma == 0.9
    
    def test_get_state_vector(self):
        """Test state vector generation."""
        pos = Position(5, 5)
        agent = RLAgent(0, pos, energy=60)
        env = Environment(width=10, height=10)
        
        # Add some entities to environment
        env.grid[3, 3] = CellType.PLANT.value
        env.grid[7, 7] = CellType.PLANT.value
        prey = Prey(1, Position(4, 4), energy=50)
        predator = Predator(2, Position(6, 6), energy=50)
        env._add_agent(prey, Position(4, 4))
        env._add_agent(predator, Position(6, 6))
        
        state = agent.get_state_vector(env)
        
        # Check state vector properties
        assert len(state) == 13  # Expected state vector length
        assert isinstance(state, np.ndarray)
        assert state.dtype == np.float32
        assert not np.isnan(state).any()
        assert not np.isinf(state).any()
        
        # Check energy normalization
        assert state[0] == 0.6  # 60/100
    
    def test_get_direction_to_nearest(self):
        """Test direction calculation to nearest entities."""
        pos = Position(5, 5)
        agent = RLAgent(0, pos)
        env = Environment(width=10, height=10)
        
        # Add plant at (7, 7)
        env.grid[7, 7] = CellType.PLANT.value
        
        # Get direction to nearest plant
        direction = agent._get_direction_to_nearest(env, CellType.PLANT, 5)
        
        assert len(direction) == 2
        assert direction[0] == 0.4  # (7-5)/5
        assert direction[1] == 0.4  # (7-5)/5
    
    def test_get_available_actions(self):
        """Test getting available actions."""
        pos = Position(5, 5)
        agent = RLAgent(0, pos)
        env = Environment(width=10, height=10)
        
        actions = agent.get_available_actions(env)
        
        # Should include current position and adjacent positions
        assert len(actions) > 0
        assert pos in actions  # Current position should be available
        
        # All actions should be valid
        for action in actions:
            assert env.is_valid_position(action)
            assert env.is_empty(action) or action == pos
    
    def test_remember_experience(self):
        """Test storing experience in memory."""
        pos = Position(5, 5)
        agent = RLAgent(0, pos)
        
        state = np.array([0.5, 0.1, 0.2, 0.3])
        action = Position(6, 6)
        reward = 10.0
        next_state = np.array([0.6, 0.2, 0.3, 0.4])
        done = False
        
        agent.remember(state, action, reward, next_state, done)
        
        assert len(agent.memory) == 1
        memory_item = agent.memory[0]
        assert memory_item[0] is state
        assert memory_item[1] == action
        assert memory_item[2] == reward
        assert memory_item[3] is next_state
        assert memory_item[4] == done

class TestQLearningAgent:
    """Test cases for the QLearningAgent class."""
    
    def test_qlearning_agent_initialization(self):
        """Test QLearningAgent initialization."""
        pos = Position(5, 5)
        agent = QLearningAgent(0, pos)
        
        assert isinstance(agent.q_table, dict)
        assert agent.learning_rate == 0.1
        assert agent.epsilon == 0.1
        assert agent.gamma == 0.95
    
    def test_state_action_key_generation(self):
        """Test state-action key generation for Q-table."""
        pos = Position(5, 5)
        agent = QLearningAgent(0, pos)
        
        state = np.array([0.5, 0.1, 0.2, 0.3])
        action = Position(6, 6)
        
        key = agent._get_state_action_key(state, action)
        
        assert isinstance(key, str)
        assert len(key) > 0
    
    def test_q_value_operations(self):
        """Test Q-value get and set operations."""
        pos = Position(5, 5)
        agent = QLearningAgent(0, pos)
        
        state = np.array([0.5, 0.1, 0.2, 0.3])
        action = Position(6, 6)
        
        # Initially should return 0.0
        assert agent._get_q_value(state, action) == 0.0
        
        # Set Q-value
        agent._set_q_value(state, action, 5.0)
        assert agent._get_q_value(state, action) == 5.0
    
    def test_get_best_action(self):
        """Test getting best action from Q-table."""
        pos = Position(5, 5)
        agent = QLearningAgent(0, pos)
        
        state = np.array([0.5, 0.1, 0.2, 0.3])
        actions = [Position(5, 5), Position(6, 6), Position(4, 4)]
        
        # Set different Q-values
        agent._set_q_value(state, Position(5, 5), 1.0)
        agent._set_q_value(state, Position(6, 6), 3.0)
        agent._set_q_value(state, Position(4, 4), 2.0)
        
        best_action = agent._get_best_action(state, actions)
        assert best_action == Position(6, 6)  # Highest Q-value
    
    def test_learning(self):
        """Test Q-learning update mechanism."""
        pos = Position(5, 5)
        agent = QLearningAgent(0, pos)
        
        # Add experiences to memory
        state = np.array([0.5, 0.1, 0.2, 0.3])
        action = Position(6, 6)
        reward = 10.0
        next_state = np.array([0.6, 0.2, 0.3, 0.4])
        done = False
        
        for _ in range(agent.batch_size):
            agent.remember(state, action, reward, next_state, done)
        
        # Learn
        agent.learn()
        
        # Q-value should have been updated
        q_value = agent._get_q_value(state, action)
        assert q_value > 0.0

class TestDQNAgent:
    """Test cases for the DQNAgent class."""
    
    def test_dqn_agent_initialization(self):
        """Test DQNAgent initialization."""
        pos = Position(5, 5)
        agent = DQNAgent(0, pos)
        
        assert agent.input_size == 13
        assert agent.output_size == 9
        assert agent.hidden_size == 64
        assert isinstance(agent.q_network, QNetwork)
        assert isinstance(agent.target_network, QNetwork)
        assert isinstance(agent.optimizer, torch.optim.Adam)
    
    def test_action_to_index_conversion(self):
        """Test action to index conversion."""
        pos = Position(5, 5)
        agent = DQNAgent(0, pos)
        
        # Test stay action
        assert agent._action_to_index(Position(5, 5)) == 0
        
        # Test directional actions
        assert agent._action_to_index(Position(4, 4)) == 1  # (-1, -1)
        assert agent._action_to_index(Position(5, 4)) == 2  # (0, -1)
        assert agent._action_to_index(Position(6, 4)) == 3  # (1, -1)
        assert agent._action_to_index(Position(6, 5)) == 4  # (1, 0)
        assert agent._action_to_index(Position(6, 6)) == 5  # (1, 1)
        assert agent._action_to_index(Position(5, 6)) == 6  # (0, 1)
        assert agent._action_to_index(Position(4, 6)) == 7  # (-1, 1)
        assert agent._action_to_index(Position(4, 5)) == 8  # (-1, 0)
    
    def test_index_to_action_conversion(self):
        """Test index to action conversion."""
        pos = Position(5, 5)
        agent = DQNAgent(0, pos)
        
        available_actions = [
            Position(5, 5), Position(4, 4), Position(6, 6)
        ]
        
        # Test stay action
        action = agent._index_to_action(0, available_actions)
        assert action == Position(5, 5)
        
        # Test directional action
        action = agent._index_to_action(1, available_actions)
        assert action == Position(4, 4)
    
    def test_get_best_action(self):
        """Test getting best action from Q-network."""
        pos = Position(5, 5)
        agent = DQNAgent(0, pos)
        
        state = np.array([0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2])
        available_actions = [Position(5, 5), Position(6, 6), Position(4, 4)]
        
        best_action = agent._get_best_action(state, available_actions)
        
        assert best_action in available_actions
    
    def test_learning(self):
        """Test DQN learning mechanism."""
        pos = Position(5, 5)
        agent = DQNAgent(0, pos)
        
        # Add experiences to memory
        state = np.array([0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2])
        action = Position(6, 6)
        reward = 10.0
        next_state = np.array([0.6, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3])
        done = False
        
        for _ in range(agent.batch_size):
            agent.remember(state, action, reward, next_state, done)
        
        # Learn
        agent.learn()
        
        # Network should have been updated
        assert agent.step_count > 0

class TestRLPrey:
    """Test cases for the RLPrey class."""
    
    def test_rl_prey_initialization(self):
        """Test RLPrey initialization."""
        pos = Position(5, 5)
        prey = RLPrey(0, pos)
        
        # Should inherit from both DQNAgent and Prey
        assert prey.cell_type == CellType.PREY
        assert prey.energy_decay_rate == 1.5
        assert prey.vision_range == 8
        assert prey.input_size == 13
        assert prey.output_size == 9
    
    def test_rl_prey_act_method(self):
        """Test RLPrey act method with learning."""
        pos = Position(5, 5)
        prey = RLPrey(0, pos)
        env = Environment(width=10, height=10)
        
        # Add some entities for state representation
        env.grid[3, 3] = CellType.PLANT.value
        predator = Predator(1, Position(6, 6), energy=50)
        env._add_agent(predator, Position(6, 6))
        
        initial_memory_size = len(prey.memory)
        
        # Act should store experience and learn
        new_pos = prey.act(env)
        
        assert new_pos is not None
        assert len(prey.memory) > initial_memory_size
    
    def test_rl_prey_reward_calculation(self):
        """Test RLPrey reward calculation."""
        pos = Position(5, 5)
        prey = RLPrey(0, pos, energy=60)
        env = Environment(width=10, height=10)
        
        # Test reward for eating plant
        plant_pos = Position(5, 5)
        env.grid[plant_pos.y, plant_pos.x] = CellType.PLANT.value
        
        reward = prey._calculate_reward(env, plant_pos)
        assert reward > 0  # Should get positive reward for eating plant
        
        # Test penalty for being near predators
        predator = Predator(1, Position(6, 6), energy=50)
        env._add_agent(predator, Position(6, 6))
        
        reward = prey._calculate_reward(env, Position(6, 6))
        assert reward < 0  # Should get negative reward for being near predator

class TestRLPredator:
    """Test cases for the RLPredator class."""
    
    def test_rl_predator_initialization(self):
        """Test RLPredator initialization."""
        pos = Position(5, 5)
        predator = RLPredator(0, pos)
        
        # Should inherit from both DQNAgent and Predator
        assert predator.cell_type == CellType.PREDATOR
        assert predator.energy_decay_rate == 2.0
        assert predator.vision_range == 10
        assert predator.input_size == 13
        assert predator.output_size == 9
    
    def test_rl_predator_act_method(self):
        """Test RLPredator act method with learning."""
        pos = Position(5, 5)
        predator = RLPredator(0, pos)
        env = Environment(width=10, height=10)
        
        # Add some entities for state representation
        prey = Prey(1, Position(6, 6), energy=50)
        env._add_agent(prey, Position(6, 6))
        
        initial_memory_size = len(predator.memory)
        
        # Act should store experience and learn
        new_pos = predator.act(env)
        
        assert new_pos is not None
        assert len(predator.memory) > initial_memory_size
    
    def test_rl_predator_reward_calculation(self):
        """Test RLPredator reward calculation."""
        pos = Position(5, 5)
        predator = RLPredator(0, pos, energy=60)
        env = Environment(width=10, height=10)
        
        # Test reward for being near prey
        prey = Prey(1, Position(6, 6), energy=50)
        env._add_agent(prey, Position(6, 6))
        
        reward = predator._calculate_reward(env, Position(6, 6))
        assert reward > 0  # Should get positive reward for being near prey
        
        # Test penalty for low energy
        predator.energy = 20
        reward = predator._calculate_reward(env, Position(5, 5))
        assert reward < 0  # Should get negative reward for low energy

class TestRLAgentIntegration:
    """Integration tests for RL agents in environment."""
    
    def test_rl_agents_in_environment(self):
        """Test RL agents functioning in environment."""
        env = Environment(width=10, height=10)
        
        # Add RL agents
        rl_prey = RLPrey(0, Position(5, 5), energy=50)
        rl_predator = RLPredator(1, Position(6, 6), energy=50)
        
        env._add_agent(rl_prey, Position(5, 5))
        env._add_agent(rl_predator, Position(6, 6))
        
        # Add some plants
        env.grid[3, 3] = CellType.PLANT.value
        env.grid[7, 7] = CellType.PLANT.value
        
        # Run a few steps
        for _ in range(5):
            env.step()
        
        # Agents should still be alive and learning
        assert rl_prey.id in env.agent_positions
        assert rl_predator.id in env.agent_positions
        assert len(rl_prey.memory) > 0
        assert len(rl_predator.memory) > 0
    
    def test_rl_agent_state_consistency(self):
        """Test that RL agent state representation is consistent."""
        pos = Position(5, 5)
        rl_prey = RLPrey(0, pos)
        env = Environment(width=10, height=10)
        
        # Get state multiple times
        state1 = rl_prey.get_state_vector(env)
        state2 = rl_prey.get_state_vector(env)
        
        # States should be identical for same environment state
        np.testing.assert_array_equal(state1, state2)
        
        # Change environment and get new state
        env.grid[3, 3] = CellType.PLANT.value
        state3 = rl_prey.get_state_vector(env)
        
        # States should be different
        assert not np.array_equal(state1, state3)

if __name__ == "__main__":
    pytest.main([__file__])
