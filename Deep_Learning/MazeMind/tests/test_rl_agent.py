"""
Tests for MazeMind - RL Agent Module
"""

import pytest
import numpy as np
import tempfile
import os
from rl_agent import QLearningAgent, MultiAgentSystem


class TestQLearningAgent:
    """Test cases for QLearningAgent class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.agent = QLearningAgent(
            learning_rate=0.1,
            discount_factor=0.9,
            epsilon=0.1,
            epsilon_decay=0.995,
            min_epsilon=0.01
        )
        
        # Create simple test maze
        self.simple_maze = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 0, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0]
        ])
        
        self.start = (1, 1)
        self.goal = (3, 3)
    
    def test_init(self):
        """Test QLearningAgent initialization."""
        assert self.agent.learning_rate == 0.1
        assert self.agent.discount_factor == 0.9
        assert self.agent.epsilon == 0.1
        assert self.agent.epsilon_decay == 0.995
        assert self.agent.min_epsilon == 0.01
        
        assert len(self.agent.actions) == 4
        assert len(self.agent.action_names) == 4
        assert len(self.agent.episode_rewards) == 0
        assert len(self.agent.episode_steps) == 0
    
    def test_get_state(self):
        """Test state representation."""
        position = (2, 2)
        state = self.agent.get_state(position, self.simple_maze)
        
        assert isinstance(state, str)
        assert len(state) > 0
        assert '_' in state  # Should contain relative position info
    
    def test_choose_action_training(self):
        """Test action selection during training."""
        state = "test_state"
        
        # Test multiple calls to ensure both exploration and exploitation occur
        actions = []
        for _ in range(100):
            action = self.agent.choose_action(state, training=True)
            actions.append(action)
        
        # Should have actions in range [0, 3]
        assert all(0 <= action < 4 for action in actions)
        
        # Should have some variety in actions (due to exploration)
        assert len(set(actions)) > 1
    
    def test_choose_action_no_training(self):
        """Test action selection without training (exploitation only)."""
        state = "test_state"
        
        # Initialize Q-values for this state
        self.agent.q_table[state][0] = 1.0
        self.agent.q_table[state][1] = 0.5
        self.agent.q_table[state][2] = 0.3
        self.agent.q_table[state][3] = 0.8
        
        # Should always choose action 0 (highest Q-value)
        for _ in range(10):
            action = self.agent.choose_action(state, training=False)
            assert action == 0
    
    def test_get_reward(self):
        """Test reward calculation."""
        # Test goal reward
        reward = self.agent.get_reward(self.goal, self.simple_maze, self.goal, 0, 100)
        assert reward == 100
        
        # Test wall penalty
        reward = self.agent.get_reward((0, 0), self.simple_maze, self.goal, 0, 100)
        assert reward == -100
        
        # Test out of bounds penalty
        reward = self.agent.get_reward((10, 10), self.simple_maze, self.goal, 0, 100)
        assert reward == -100
        
        # Test timeout penalty
        reward = self.agent.get_reward(self.start, self.simple_maze, self.goal, 100, 100)
        assert reward == -50
        
        # Test distance-based reward
        reward = self.agent.get_reward(self.start, self.simple_maze, self.goal, 0, 100)
        assert reward < 0  # Should be negative (distance penalty)
    
    def test_update_q_value(self):
        """Test Q-value update."""
        state = "test_state"
        action = 0
        reward = 10
        next_state = "next_state"
        
        # Initialize Q-values
        self.agent.q_table[state][action] = 0.0
        self.agent.q_table[next_state][0] = 5.0
        self.agent.q_table[next_state][1] = 3.0
        
        # Update Q-value
        self.agent.update_q_value(state, action, reward, next_state)
        
        # Check that Q-value was updated
        assert self.agent.q_table[state][action] != 0.0
        
        # Q-value should be positive (reward + discount * max_next_q)
        assert self.agent.q_table[state][action] > 0
    
    def test_train_episode(self):
        """Test episode training."""
        result = self.agent.train_episode(self.simple_maze, self.start, self.goal, max_steps=100)
        
        assert 'success' in result
        assert 'total_reward' in result
        assert 'steps' in result
        assert 'path' in result
        assert 'final_position' in result
        
        assert isinstance(result['success'], bool)
        assert isinstance(result['total_reward'], (int, float))
        assert isinstance(result['steps'], int)
        assert isinstance(result['path'], list)
        assert isinstance(result['final_position'], tuple)
        
        assert result['steps'] > 0
        assert len(result['path']) > 0
        assert result['path'][0] == self.start
    
    def test_solve_maze(self):
        """Test maze solving with trained agent."""
        # Train agent first
        self.agent.train_episode(self.simple_maze, self.start, self.goal, max_steps=100)
        
        # Solve maze
        result = self.agent.solve_maze(self.simple_maze, self.start, self.goal, max_steps=100)
        
        assert 'success' in result
        assert 'steps' in result
        assert 'path' in result
        assert 'final_position' in result
        
        assert isinstance(result['success'], bool)
        assert isinstance(result['steps'], int)
        assert isinstance(result['path'], list)
        assert isinstance(result['final_position'], tuple)
        
        assert result['steps'] > 0
        assert len(result['path']) > 0
        assert result['path'][0] == self.start
    
    def test_epsilon_decay(self):
        """Test epsilon decay during training."""
        initial_epsilon = self.agent.epsilon
        
        # Train for several episodes
        for _ in range(10):
            self.agent.train_episode(self.simple_maze, self.start, self.goal, max_steps=50)
        
        # Epsilon should have decreased
        assert self.agent.epsilon < initial_epsilon
        
        # Epsilon should not go below minimum
        assert self.agent.epsilon >= self.agent.min_epsilon
    
    def test_get_training_stats(self):
        """Test training statistics."""
        # No training yet
        stats = self.agent.get_training_stats()
        assert stats == {}
        
        # Train for some episodes
        for _ in range(5):
            self.agent.train_episode(self.simple_maze, self.start, self.goal, max_steps=50)
        
        stats = self.agent.get_training_stats()
        
        assert 'total_episodes' in stats
        assert 'average_reward' in stats
        assert 'average_steps' in stats
        assert 'current_epsilon' in stats
        assert 'q_table_size' in stats
        
        assert stats['total_episodes'] == 5
        assert stats['current_epsilon'] >= self.agent.min_epsilon
        assert stats['q_table_size'] > 0
    
    def test_save_load_q_table(self):
        """Test Q-table save and load functionality."""
        # Train agent
        self.agent.train_episode(self.simple_maze, self.start, self.goal, max_steps=50)
        
        # Save Q-table
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
            filename = tmp_file.name
        
        try:
            self.agent.save_q_table(filename)
            
            # Create new agent and load Q-table
            new_agent = QLearningAgent()
            new_agent.load_q_table(filename)
            
            # Check that non-zero Q-values are preserved
            for state in self.agent.q_table:
                for action in self.agent.q_table[state]:
                    original_value = self.agent.q_table[state][action]
                    if original_value != 0.0:  # Only check non-zero values
                        loaded_value = new_agent.q_table[state][action]
                        assert abs(original_value - loaded_value) < 1e-10
        
        finally:
            # Clean up
            if os.path.exists(filename):
                os.unlink(filename)
    
    def test_episode_statistics(self):
        """Test episode statistics tracking."""
        # Train for multiple episodes
        for _ in range(3):
            result = self.agent.train_episode(self.simple_maze, self.start, self.goal, max_steps=50)
        
        # Check that statistics are tracked
        assert len(self.agent.episode_rewards) == 3
        assert len(self.agent.episode_steps) == 3
        
        # All should be numeric
        assert all(isinstance(r, (int, float)) for r in self.agent.episode_rewards)
        assert all(isinstance(s, int) for s in self.agent.episode_steps)
    
    def test_action_validity(self):
        """Test that actions are valid."""
        state = "test_state"
        
        for _ in range(100):
            action = self.agent.choose_action(state, training=True)
            assert 0 <= action < len(self.agent.actions)
            
            # Check that action corresponds to valid direction
            dx, dy = self.agent.actions[action]
            assert isinstance(dx, int)
            assert isinstance(dy, int)
            assert abs(dx) + abs(dy) == 1  # Manhattan distance = 1


class TestMultiAgentSystem:
    """Test cases for MultiAgentSystem class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.system = MultiAgentSystem()
        
        # Create test agents
        self.agent1 = QLearningAgent(learning_rate=0.1, epsilon=0.2)
        self.agent2 = QLearningAgent(learning_rate=0.2, epsilon=0.1)
        
        # Create test maze
        self.test_maze = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 0, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0]
        ])
        
        self.start = (1, 1)
        self.goal = (3, 3)
    
    def test_init(self):
        """Test MultiAgentSystem initialization."""
        assert len(self.system.agents) == 0
    
    def test_add_agent(self):
        """Test adding agents to the system."""
        self.system.add_agent("agent1", self.agent1)
        self.system.add_agent("agent2", self.agent2)
        
        assert len(self.system.agents) == 2
        assert "agent1" in self.system.agents
        assert "agent2" in self.system.agents
        assert self.system.agents["agent1"] == self.agent1
        assert self.system.agents["agent2"] == self.agent2
    
    def test_train_all_agents(self):
        """Test training all agents."""
        self.system.add_agent("agent1", self.agent1)
        self.system.add_agent("agent2", self.agent2)
        
        results = self.system.train_all_agents(self.test_maze, self.start, self.goal, episodes=5)
        
        assert "agent1" in results
        assert "agent2" in results
        
        assert len(results["agent1"]) == 5
        assert len(results["agent2"]) == 5
        
        # Check that each episode result has required fields
        for agent_results in results.values():
            for episode_result in agent_results:
                assert 'success' in episode_result
                assert 'total_reward' in episode_result
                assert 'steps' in episode_result
                assert 'path' in episode_result
                assert 'final_position' in episode_result
    
    def test_compare_agents(self):
        """Test agent comparison."""
        self.system.add_agent("agent1", self.agent1)
        self.system.add_agent("agent2", self.agent2)
        
        # Train agents first
        self.system.train_all_agents(self.test_maze, self.start, self.goal, episodes=3)
        
        # Compare agents
        results = self.system.compare_agents(self.test_maze, self.start, self.goal)
        
        assert "agent1" in results
        assert "agent2" in results
        
        # Check that each result has required fields
        for agent_result in results.values():
            assert 'success' in agent_result
            assert 'steps' in agent_result
            assert 'path' in agent_result
            assert 'final_position' in agent_result
    
    def test_empty_system(self):
        """Test behavior with empty system."""
        # Training empty system should return empty results
        results = self.system.train_all_agents(self.test_maze, self.start, self.goal, episodes=5)
        assert results == {}
        
        # Comparison with empty system should return empty results
        results = self.system.compare_agents(self.test_maze, self.start, self.goal)
        assert results == {}


if __name__ == "__main__":
    pytest.main([__file__])
