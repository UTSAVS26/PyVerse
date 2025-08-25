"""
Tests for the agent module
"""

import pytest
import numpy as np
import torch
import tempfile
import os

# Import the modules to test
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import PaintingEnvironment, PaintingActor, PaintingCritic, PaintingAgent


class TestPaintingEnvironment:
    """Test cases for the PaintingEnvironment class"""
    
    def test_environment_initialization(self):
        """Test environment initialization"""
        env = PaintingEnvironment(canvas_width=400, canvas_height=300, max_strokes=20)
        assert env.canvas_width == 400
        assert env.canvas_height == 300
        assert env.max_strokes == 20
        assert env.current_stroke_count == 0
        assert env.episode_reward == 0.0
    
    def test_environment_reset(self):
        """Test environment reset"""
        env = PaintingEnvironment()
        
        # Apply some strokes
        action = np.array([0.0, 0.5, 0.5, 0.0, 1.0, 0.0, 0.0, 0.5])  # Line stroke
        env.step(action)
        
        # Reset environment
        state = env.reset()
        
        assert env.current_stroke_count == 0
        assert env.episode_reward == 0.0
        assert len(env.canvas.stroke_history) == 0
        assert isinstance(state, np.ndarray)
    
    def test_get_state(self):
        """Test state representation"""
        env = PaintingEnvironment(canvas_width=100, canvas_height=100)
        state = env._get_state()
        
        # State should be flattened canvas + metadata
        expected_size = 100 * 100 * 3 + 3  # image + coverage, diversity, stroke_count
        assert state.shape == (expected_size,)
        assert state.dtype == np.float32
    
    def test_step_line_stroke(self):
        """Test taking a step with line stroke"""
        env = PaintingEnvironment(canvas_width=200, canvas_height=200)
        
        # Action: [stroke_type, x, y, angle, color_r, color_g, color_b, thickness]
        action = np.array([0.0, 0.5, 0.5, 0.25, 1.0, 0.0, 0.0, 0.3])  # Red line
        
        state = env.reset()
        next_state, reward, done, info = env.step(action)
        
        assert isinstance(next_state, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        assert env.current_stroke_count == 1
        assert len(env.canvas.stroke_history) == 1
        assert info['stroke_type'] == 'line'
        assert info['success'] == True
    
    def test_step_curve_stroke(self):
        """Test taking a step with curve stroke"""
        env = PaintingEnvironment(canvas_width=200, canvas_height=200)
        
        # Action: curve stroke
        action = np.array([0.25, 0.3, 0.7, 0.5, 0.0, 1.0, 0.0, 0.2])  # Green curve
        
        state = env.reset()
        next_state, reward, done, info = env.step(action)
        
        assert info['stroke_type'] == 'curve'
        assert info['success'] == True
    
    def test_step_dot_stroke(self):
        """Test taking a step with dot stroke"""
        env = PaintingEnvironment(canvas_width=200, canvas_height=200)
        
        # Action: dot stroke
        action = np.array([0.5, 0.6, 0.4, 0.0, 0.0, 0.0, 1.0, 0.1])  # Blue dot
        
        state = env.reset()
        next_state, reward, done, info = env.step(action)
        
        assert info['stroke_type'] == 'dot'
        assert info['success'] == True
    
    def test_step_splash_stroke(self):
        """Test taking a step with splash stroke"""
        env = PaintingEnvironment(canvas_width=200, canvas_height=200)
        
        # Action: splash stroke
        action = np.array([0.75, 0.8, 0.2, 0.0, 1.0, 1.0, 0.0, 0.1])  # Yellow splash
        
        state = env.reset()
        next_state, reward, done, info = env.step(action)
        
        assert info['stroke_type'] == 'splash'
        assert info['success'] == True
    
    def test_episode_completion(self):
        """Test episode completion after max strokes"""
        env = PaintingEnvironment(canvas_width=100, canvas_height=100, max_strokes=3)
        
        state = env.reset()
        action = np.array([0.0, 0.5, 0.5, 0.0, 1.0, 0.0, 0.0, 0.5])
        
        # Take 3 steps
        for i in range(3):
            next_state, reward, done, info = env.step(action)
            if i < 2:
                assert done == False
            else:
                assert done == True
        
        assert env.current_stroke_count == 3
    
    def test_reward_calculation(self):
        """Test reward calculation"""
        env = PaintingEnvironment(canvas_width=100, canvas_height=100)
        
        state = env.reset()
        
        # Apply a stroke
        action = np.array([0.0, 0.5, 0.5, 0.0, 1.0, 0.0, 0.0, 0.5])
        next_state, reward, done, info = env.step(action)
        
        # Reward should be positive for successful stroke
        assert reward > 0.0
        
        # Apply another stroke with different color
        action = np.array([0.5, 0.3, 0.7, 0.0, 0.0, 1.0, 0.0, 0.5])
        next_state, reward, done, info = env.step(action)
        
        # Reward should be higher due to color diversity
        assert reward > 0.0
    
    def test_coverage_tracking(self):
        """Test coverage tracking in info"""
        env = PaintingEnvironment(canvas_width=100, canvas_height=100)
        
        state = env.reset()
        action = np.array([0.5, 0.5, 0.5, 0.0, 1.0, 0.0, 0.0, 0.5])  # Large dot
        
        next_state, reward, done, info = env.step(action)
        
        assert 'coverage' in info
        assert info['coverage'] > 0.0
        assert info['coverage'] <= 1.0
    
    def test_color_diversity_tracking(self):
        """Test color diversity tracking in info"""
        env = PaintingEnvironment(canvas_width=100, canvas_height=100)
        
        state = env.reset()
        
        # Apply strokes with different colors
        colors = [
            [1.0, 0.0, 0.0],  # Red
            [0.0, 1.0, 0.0],  # Green
            [0.0, 0.0, 1.0]   # Blue
        ]
        
        for i, color in enumerate(colors):
            action = np.array([0.0, 0.5, 0.5, 0.0] + color + [0.5])
            next_state, reward, done, info = env.step(action)
        
        # Color diversity should be 1.0 (all different colors)
        assert info['color_diversity'] == 1.0


class TestPaintingActor:
    """Test cases for the PaintingActor class"""
    
    def test_actor_initialization(self):
        """Test actor network initialization"""
        actor = PaintingActor(state_dim=100, action_dim=8)
        
        assert isinstance(actor.fc1, torch.nn.Linear)
        assert isinstance(actor.fc2, torch.nn.Linear)
        assert isinstance(actor.fc3, torch.nn.Linear)
        
        assert actor.fc1.in_features == 100
        assert actor.fc3.out_features == 8
    
    def test_actor_forward(self):
        """Test actor forward pass"""
        actor = PaintingActor(state_dim=100, action_dim=8)
        
        # Create dummy state
        state = torch.randn(1, 100)
        
        # Forward pass
        action = actor(state)
        
        assert action.shape == (1, 8)
        assert torch.all(action >= 0.0) and torch.all(action <= 1.0)  # Sigmoid output
    
    def test_actor_batch_forward(self):
        """Test actor forward pass with batch"""
        actor = PaintingActor(state_dim=100, action_dim=8)
        
        # Create batch of states
        batch_size = 5
        states = torch.randn(batch_size, 100)
        
        # Forward pass
        actions = actor(states)
        
        assert actions.shape == (batch_size, 8)
        assert torch.all(actions >= 0.0) and torch.all(actions <= 1.0)


class TestPaintingCritic:
    """Test cases for the PaintingCritic class"""
    
    def test_critic_initialization(self):
        """Test critic network initialization"""
        critic = PaintingCritic(state_dim=100, action_dim=8)
        
        assert isinstance(critic.fc1, torch.nn.Linear)
        assert isinstance(critic.fc2, torch.nn.Linear)
        assert isinstance(critic.fc3, torch.nn.Linear)
        
        assert critic.fc1.in_features == 108  # state_dim + action_dim
        assert critic.fc3.out_features == 1
    
    def test_critic_forward(self):
        """Test critic forward pass"""
        critic = PaintingCritic(state_dim=100, action_dim=8)
        
        # Create dummy state and action
        state = torch.randn(1, 100)
        action = torch.randn(1, 8)
        
        # Forward pass
        q_value = critic(state, action)
        
        assert q_value.shape == (1, 1)
        assert isinstance(q_value, torch.Tensor)
    
    def test_critic_batch_forward(self):
        """Test critic forward pass with batch"""
        critic = PaintingCritic(state_dim=100, action_dim=8)
        
        # Create batch of states and actions
        batch_size = 5
        states = torch.randn(batch_size, 100)
        actions = torch.randn(batch_size, 8)
        
        # Forward pass
        q_values = critic(states, actions)
        
        assert q_values.shape == (batch_size, 1)


class TestPaintingAgent:
    """Test cases for the PaintingAgent class"""
    
    def test_agent_initialization(self):
        """Test agent initialization"""
        agent = PaintingAgent(state_dim=100, action_dim=8)
        
        assert agent.state_dim == 100
        assert agent.action_dim == 8
        assert agent.gamma == 0.99
        assert agent.tau == 0.005
        assert len(agent.memory) == 0
        assert agent.noise_std == 0.1
    
    def test_select_action(self):
        """Test action selection"""
        agent = PaintingAgent(state_dim=100, action_dim=8)
        
        # Create dummy state
        state = np.random.randn(100).astype(np.float32)
        
        # Select action without noise
        action = agent.select_action(state, add_noise=False)
        
        assert action.shape == (8,)
        assert np.all(action >= 0.0) and np.all(action <= 1.0)
        
        # Select action with noise
        action_with_noise = agent.select_action(state, add_noise=True)
        
        assert action_with_noise.shape == (8,)
        assert np.all(action_with_noise >= 0.0) and np.all(action_with_noise <= 1.0)
    
    def test_store_transition(self):
        """Test storing transitions in memory"""
        agent = PaintingAgent(state_dim=100, action_dim=8)
        
        # Create dummy transition
        state = np.random.randn(100)
        action = np.random.randn(8)
        reward = 1.5
        next_state = np.random.randn(100)
        done = False
        
        # Store transition
        agent.store_transition(state, action, reward, next_state, done)
        
        assert len(agent.memory) == 1
        
        # Store more transitions
        for _ in range(5):
            agent.store_transition(state, action, reward, next_state, done)
        
        assert len(agent.memory) == 6
    
    def test_train_with_insufficient_memory(self):
        """Test training with insufficient memory"""
        agent = PaintingAgent(state_dim=100, action_dim=8)
        
        # Try to train without enough memory
        result = agent.train(batch_size=64)
        assert result is None
    
    def test_train_with_sufficient_memory(self):
        """Test training with sufficient memory"""
        agent = PaintingAgent(state_dim=100, action_dim=8)
        
        # Fill memory with transitions
        state = np.random.randn(100)
        action = np.random.randn(8)
        reward = 1.0
        next_state = np.random.randn(100)
        done = False
        
        for _ in range(100):  # More than batch_size
            agent.store_transition(state, action, reward, next_state, done)
        
        # Train
        result = agent.train(batch_size=64)
        
        assert result is not None
        assert 'critic_loss' in result
        assert 'actor_loss' in result
        assert isinstance(result['critic_loss'], float)
        assert isinstance(result['actor_loss'], float)
    
    def test_save_and_load_model(self):
        """Test saving and loading models"""
        agent = PaintingAgent(state_dim=100, action_dim=8)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp_file:
            model_path = tmp_file.name
        
        try:
            # Save model
            agent.save_model(model_path)
            assert os.path.exists(model_path)
            assert os.path.getsize(model_path) > 0
            
            # Create new agent and load model
            new_agent = PaintingAgent(state_dim=100, action_dim=8)
            new_agent.load_model(model_path)
            
            # Test that models are the same
            state = np.random.randn(100).astype(np.float32)
            action1 = agent.select_action(state, add_noise=False)
            action2 = new_agent.select_action(state, add_noise=False)
            
            np.testing.assert_array_almost_equal(action1, action2)
            
        finally:
            # Clean up
            if os.path.exists(model_path):
                os.unlink(model_path)
    
    def test_target_network_update(self):
        """Test target network update"""
        agent = PaintingAgent(state_dim=100, action_dim=8, tau=0.5)  # Use very large tau for testing
        
        # Get initial target network parameters
        initial_target_params = []
        for param in agent.target_actor.parameters():
            initial_target_params.append(param.data.clone())
        
        # Train to trigger target network update
        state = np.random.randn(100)
        action = np.random.randn(8)
        reward = 1.0
        next_state = np.random.randn(100)
        done = False
        
        for _ in range(200):
            agent.store_transition(state, action, reward, next_state, done)
        
        # Train multiple times to ensure target network updates
        for _ in range(20):
            agent.train(batch_size=64)
        
        # Check that target networks were updated
        updated_target_params = []
        for param in agent.target_actor.parameters():
            updated_target_params.append(param.data.clone())
        
        # Parameters should be different with large tau
        params_changed = False
        for initial, updated in zip(initial_target_params, updated_target_params):
            if not torch.allclose(initial, updated, atol=1e-1):
                params_changed = True
                break
        
        assert params_changed


class TestIntegration:
    """Integration tests for the complete system"""
    
    def test_environment_agent_integration(self):
        """Test integration between environment and agent"""
        # Create environment and agent
        env = PaintingEnvironment(canvas_width=100, canvas_height=100, max_strokes=5)
        state_dim = 100 * 100 * 3 + 3  # canvas + metadata
        action_dim = 8
        agent = PaintingAgent(state_dim=state_dim, action_dim=action_dim)
        
        # Run a few episodes
        for episode in range(2):
            state = env.reset()
            
            for step in range(5):
                # Select action
                action = agent.select_action(state, add_noise=True)
                
                # Take step
                next_state, reward, done, info = env.step(action)
                
                # Store transition
                agent.store_transition(state, action, reward, next_state, done)
                
                # Update state
                state = next_state
                
                if done:
                    break
            
            # Train agent
            if len(agent.memory) > 64:
                agent.train(batch_size=64)
        
        # Verify that training occurred
        assert len(agent.memory) > 0
    
    def test_full_training_cycle(self):
        """Test a complete training cycle"""
        # Create environment and agent
        env = PaintingEnvironment(canvas_width=50, canvas_height=50, max_strokes=3)
        state_dim = 50 * 50 * 3 + 3
        action_dim = 8
        agent = PaintingAgent(state_dim=state_dim, action_dim=action_dim)
        
        # Run training cycle
        state = env.reset()
        total_reward = 0.0
        
        for step in range(3):
            # Select action
            action = agent.select_action(state, add_noise=True)
            
            # Take step
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # Update state
            state = next_state
            
            if done:
                break
        
        # Train agent
        if len(agent.memory) >= 3:
            result = agent.train(batch_size=3)
            assert result is not None
        
        # Verify results
        assert total_reward > 0.0  # Should have positive reward
        assert len(env.canvas.stroke_history) > 0  # Should have applied strokes


if __name__ == "__main__":
    pytest.main([__file__])
