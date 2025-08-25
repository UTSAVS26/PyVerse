import pytest
import numpy as np
import torch
import tempfile
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from agents.qlearning_agent import QLearningAgent
from agents.dqn_agent import DQNAgent
from environment.snake_env import SnakeEnv

class TestQLearningAgent:
    """Test cases for Q-Learning agent"""
    
    def setup_method(self):
        """Setup method called before each test"""
        self.state_size = 11
        self.action_size = 3
        self.agent = QLearningAgent(self.state_size, self.action_size)
        self.env = SnakeEnv(grid_size=5, render_mode="rgb_array")
    
    def teardown_method(self):
        """Teardown method called after each test"""
        self.env.close()
    
    def test_initialization(self):
        """Test agent initialization"""
        assert self.agent.state_size == self.state_size
        assert self.agent.action_size == self.action_size
        assert self.agent.epsilon == 1.0
        assert self.agent.epsilon_min == 0.01
        assert self.agent.epsilon_decay == 0.995
        assert self.agent.learning_rate == 0.1
        assert self.agent.discount_factor == 0.95
        assert isinstance(self.agent.q_table, dict)
    
    def test_state_discretization(self):
        """Test state discretization"""
        # Create a sample state
        state = np.array([0.5, 0.3, 0.8, 0.2, 1, 0, 0, 0, 0, 1, 0], dtype=np.float32)
        
        discretized = self.agent._discretize_state(state)
        
        # Check discretized state components
        assert len(discretized) == 8
        assert discretized[0] == 2  # head_x (0.5 * 4 = 2)
        assert discretized[1] == 1  # head_y (0.3 * 4 = 1)
        assert discretized[2] == 3  # food_x (0.8 * 4 = 3)
        assert discretized[3] == 0  # food_y (0.2 * 4 = 0)
        assert discretized[4] == 0  # direction (argmax of [1,0,0,0])
        assert discretized[5] == 0  # danger_left
        assert discretized[6] == 1  # danger_straight
        assert discretized[7] == 0  # danger_right
    
    def test_action_selection(self):
        """Test action selection"""
        state = np.random.random(self.state_size).astype(np.float32)
        
        # Test exploration (high epsilon)
        self.agent.epsilon = 1.0
        actions = []
        for _ in range(100):
            action = self.agent.act(state)
            actions.append(action)
            assert 0 <= action < self.action_size
        
        # Should have some variety in actions due to exploration
        assert len(set(actions)) > 1
        
        # Test exploitation (low epsilon)
        self.agent.epsilon = 0.0
        action = self.agent.act(state)
        assert 0 <= action < self.action_size
    
    def test_q_value_retrieval(self):
        """Test Q-value retrieval"""
        state = (1, 1, 2, 2, 0, 0, 0, 0)
        
        # Initially no Q-values
        q_values = self.agent._get_q_values(state)
        assert np.array_equal(q_values, np.zeros(self.action_size))
        
        # Add some Q-values
        self.agent.q_table[(state, 0)] = 1.5
        self.agent.q_table[(state, 1)] = 2.0
        self.agent.q_table[(state, 2)] = 0.5
        
        q_values = self.agent._get_q_values(state)
        assert q_values[0] == 1.5
        assert q_values[1] == 2.0
        assert q_values[2] == 0.5
    
    def test_training(self):
        """Test agent training"""
        state = np.random.random(self.state_size).astype(np.float32)
        action = 1
        reward = 10.0
        next_state = np.random.random(self.state_size).astype(np.float32)
        done = False
        
        # Train agent
        self.agent.train(state, action, reward, next_state, done)
        
        # Check Q-table was updated
        discretized_state = self.agent._discretize_state(state)
        state_action = (discretized_state, action)
        assert state_action in self.agent.q_table
        assert self.agent.q_table[state_action] > 0
    
    def test_epsilon_decay(self):
        """Test epsilon decay"""
        initial_epsilon = self.agent.epsilon
        
        # Update epsilon
        self.agent.update_epsilon()
        
        # Check epsilon decreased
        assert self.agent.epsilon < initial_epsilon
        assert self.agent.epsilon >= self.agent.epsilon_min
    
    def test_save_load(self):
        """Test model saving and loading"""
        # Add some Q-values
        state = (1, 1, 2, 2, 0, 0, 0, 0)
        self.agent.q_table[(state, 0)] = 1.5
        self.agent.q_table[(state, 1)] = 2.0
        
        # Save model
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            model_path = f.name
        
        try:
            self.agent.save(model_path)
            
            # Create new agent and load
            new_agent = QLearningAgent(self.state_size, self.action_size)
            new_agent.load(model_path)
            
            # Check Q-table was loaded correctly
            assert new_agent.q_table[(state, 0)] == 1.5
            assert new_agent.q_table[(state, 1)] == 2.0
            
        finally:
            os.unlink(model_path)
    
    def test_get_info(self):
        """Test agent info retrieval"""
        info = self.agent.get_info()
        
        assert info['type'] == 'QLearningAgent'
        assert info['state_size'] == self.state_size
        assert info['action_size'] == self.action_size
        assert info['epsilon'] == self.agent.epsilon
        assert info['learning_rate'] == self.agent.learning_rate
        assert info['discount_factor'] == self.agent.discount_factor
        assert 'q_table_size' in info

class TestDQNAgent:
    """Test cases for DQN agent"""
    
    def setup_method(self):
        """Setup method called before each test"""
        self.state_size = 11
        self.action_size = 3
        self.agent = DQNAgent(self.state_size, self.action_size)
        self.env = SnakeEnv(grid_size=5, render_mode="rgb_array")
    
    def teardown_method(self):
        """Teardown method called after each test"""
        self.env.close()
    
    def test_initialization(self):
        """Test agent initialization"""
        assert self.agent.state_size == self.state_size
        assert self.agent.action_size == self.action_size
        assert self.agent.epsilon == 1.0
        assert self.agent.epsilon_min == 0.01
        assert self.agent.epsilon_decay == 0.995
        assert self.agent.learning_rate == 0.001
        assert self.agent.discount_factor == 0.95
        assert self.agent.memory_size == 10000
        assert self.agent.batch_size == 32
        assert self.agent.target_update == 100
        assert len(self.agent.memory) == 0
        assert self.agent.train_step == 0
    
    def test_action_selection(self):
        """Test action selection"""
        state = np.random.random(self.state_size).astype(np.float32)
        
        # Test exploration (high epsilon)
        self.agent.epsilon = 1.0
        actions = []
        for _ in range(100):
            action = self.agent.act(state)
            actions.append(action)
            assert 0 <= action < self.action_size
        
        # Should have some variety in actions due to exploration
        assert len(set(actions)) > 1
        
        # Test exploitation (low epsilon)
        self.agent.epsilon = 0.0
        action = self.agent.act(state)
        assert 0 <= action < self.action_size
    
    def test_experience_replay(self):
        """Test experience replay buffer"""
        state = np.random.random(self.state_size).astype(np.float32)
        action = 1
        reward = 10.0
        next_state = np.random.random(self.state_size).astype(np.float32)
        done = False
        
        # Add experience
        self.agent.remember(state, action, reward, next_state, done)
        
        # Check memory
        assert len(self.agent.memory) == 1
        assert self.agent.memory[0] == (state, action, reward, next_state, done)
    
    def test_training_without_enough_samples(self):
        """Test training when not enough samples in memory"""
        state = np.random.random(self.state_size).astype(np.float32)
        action = 1
        reward = 10.0
        next_state = np.random.random(self.state_size).astype(np.float32)
        done = False
        
        # Train with insufficient samples
        self.agent.train(state, action, reward, next_state, done)
        
        # Should not crash, just store experience
        assert len(self.agent.memory) == 1
    
    def test_training_with_enough_samples(self):
        """Test training with enough samples"""
        # Fill memory with enough samples
        for _ in range(self.agent.batch_size):
            state = np.random.random(self.state_size).astype(np.float32)
            action = np.random.randint(0, self.action_size)
            reward = np.random.random() * 20 - 10
            next_state = np.random.random(self.state_size).astype(np.float32)
            done = np.random.choice([True, False])
            self.agent.remember(state, action, reward, next_state, done)
        
        # Train
        state = np.random.random(self.state_size).astype(np.float32)
        action = 1
        reward = 10.0
        next_state = np.random.random(self.state_size).astype(np.float32)
        done = False
        
        self.agent.train(state, action, reward, next_state, done)
        
        # Should have trained
        assert self.agent.train_step > 0
    
    def test_epsilon_decay(self):
        """Test epsilon decay"""
        initial_epsilon = self.agent.epsilon
        
        # Update epsilon
        self.agent.update_epsilon()
        
        # Check epsilon decreased
        assert self.agent.epsilon < initial_epsilon
        assert self.agent.epsilon >= self.agent.epsilon_min
    
    def test_save_load(self):
        """Test model saving and loading"""
        # Train a bit first
        for _ in range(self.agent.batch_size):
            state = np.random.random(self.state_size).astype(np.float32)
            action = np.random.randint(0, self.action_size)
            reward = np.random.random() * 20 - 10
            next_state = np.random.random(self.state_size).astype(np.float32)
            done = np.random.choice([True, False])
            self.agent.remember(state, action, reward, next_state, done)
        
        # Train once
        state = np.random.random(self.state_size).astype(np.float32)
        action = 1
        reward = 10.0
        next_state = np.random.random(self.state_size).astype(np.float32)
        done = False
        self.agent.train(state, action, reward, next_state, done)
        
        # Save model
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as f:
            model_path = f.name
        
        try:
            self.agent.save(model_path)
            
            # Create new agent and load
            new_agent = DQNAgent(self.state_size, self.action_size)
            new_agent.load(model_path)
            
            # Check some properties were loaded
            assert new_agent.epsilon == self.agent.epsilon
            assert new_agent.train_step == self.agent.train_step
            
        finally:
            os.unlink(model_path)
    
    def test_get_info(self):
        """Test agent info retrieval"""
        info = self.agent.get_info()
        
        assert info['type'] == 'DQNAgent'
        assert info['state_size'] == self.state_size
        assert info['action_size'] == self.action_size
        assert info['epsilon'] == self.agent.epsilon
        assert info['learning_rate'] == self.agent.learning_rate
        assert info['discount_factor'] == self.agent.discount_factor
        assert info['memory_size'] == 0  # Initially empty
        assert info['batch_size'] == self.agent.batch_size
        assert info['target_update'] == self.agent.target_update
        assert info['train_step'] == self.agent.train_step
        assert 'device' in info
    
    def test_neural_network_architecture(self):
        """Test neural network architecture"""
        # Test forward pass
        state = np.random.random(self.state_size).astype(np.float32)
        state_tensor = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.agent.device
        ).unsqueeze(0)

        q_values = self.agent.q_network(state_tensor).detach().to('cpu')

        # Check output shape
        assert q_values.shape == (1, self.action_size)
        assert q_values.dtype == torch.float32
    
    def test_target_network_update(self):
        """Test target network update"""
        # Train enough to trigger target update
        for _ in range(self.agent.target_update + 1):
            # Fill memory
            for _ in range(self.agent.batch_size):
                state = np.random.random(self.state_size).astype(np.float32)
                action = np.random.randint(0, self.action_size)
                reward = np.random.random() * 20 - 10
                next_state = np.random.random(self.state_size).astype(np.float32)
                done = np.random.choice([True, False])
                self.agent.remember(state, action, reward, next_state, done)
            
            # Train
            state = np.random.random(self.state_size).astype(np.float32)
            action = 1
            reward = 10.0
            next_state = np.random.random(self.state_size).astype(np.float32)
            done = False
            self.agent.train(state, action, reward, next_state, done)
        
        # Should have updated target network
        assert self.agent.train_step >= self.agent.target_update
