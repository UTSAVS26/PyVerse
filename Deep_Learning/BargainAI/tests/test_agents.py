import pytest
import numpy as np
import torch
from agents import RuleBasedAgent, RLAgent, Personality, QNetwork
from environment import NegotiationEnvironment, NegotiationState, ActionType

class TestRuleBasedAgent:
    """Test cases for rule-based agents"""
    
    def setup_method(self):
        """Set up test environment and agents"""
        self.env = NegotiationEnvironment(
            max_rounds=10,
            item_value=100.0,
            buyer_budget=120.0,
            seller_cost=60.0
        )
        self.state = self.env.reset()
    
    def test_agent_initialization(self):
        """Test agent initialization"""
        buyer = RuleBasedAgent("buyer", Personality.COOPERATIVE)
        seller = RuleBasedAgent("seller", Personality.AGGRESSIVE)
        
        assert buyer.agent_type == "buyer"
        assert buyer.personality == Personality.COOPERATIVE
        assert seller.agent_type == "seller"
        assert seller.personality == Personality.AGGRESSIVE
    
    def test_cooperative_buyer_strategy(self):
        """Test cooperative buyer strategy"""
        buyer = RuleBasedAgent("buyer", Personality.COOPERATIVE)
        
        # Test with reasonable seller offer
        self.state.last_seller_offer = 100.0  # Within budget
        action = buyer._cooperative_strategy(self.state, [])
        assert action['action_type'] == ActionType.ACCEPT
        assert action['price'] == 100.0
        
        # Test with no seller offer - should make fair offer
        self.state.last_seller_offer = None
        action = buyer._cooperative_strategy(self.state, [])
        assert action['action_type'] == ActionType.OFFER
        expected_fair_price = (60.0 + 120.0) / 2  # 90.0
        assert action['price'] == expected_fair_price
    
    def test_cooperative_seller_strategy(self):
        """Test cooperative seller strategy"""
        seller = RuleBasedAgent("seller", Personality.COOPERATIVE)
        
        # Test with reasonable buyer offer
        self.state.last_buyer_offer = 80.0  # Above cost
        action = seller._cooperative_strategy(self.state, [])
        assert action['action_type'] == ActionType.ACCEPT
        assert action['price'] == 80.0
        
        # Test with no buyer offer - should make fair offer
        self.state.last_buyer_offer = None
        action = seller._cooperative_strategy(self.state, [])
        assert action['action_type'] == ActionType.OFFER
        expected_fair_price = (60.0 + 100.0) / 2  # 80.0
        assert action['price'] == expected_fair_price
    
    def test_aggressive_buyer_strategy(self):
        """Test aggressive buyer strategy"""
        buyer = RuleBasedAgent("buyer", Personality.AGGRESSIVE)
        
        # Test with very good seller offer
        self.state.last_seller_offer = 65.0  # Very close to cost
        action = buyer._aggressive_strategy(self.state, [])
        assert action['action_type'] == ActionType.ACCEPT
        assert action['price'] == 65.0
        
        # Test with no seller offer - should make low offer
        self.state.last_seller_offer = None
        action = buyer._aggressive_strategy(self.state, [])
        assert action['action_type'] == ActionType.OFFER
        expected_aggressive_price = 60.0 * 1.05  # 63.0
        assert action['price'] == expected_aggressive_price
    
    def test_aggressive_seller_strategy(self):
        """Test aggressive seller strategy"""
        seller = RuleBasedAgent("seller", Personality.AGGRESSIVE)
        
        # Test with high buyer offer
        self.state.last_buyer_offer = 95.0  # Close to item value
        action = seller._aggressive_strategy(self.state, [])
        assert action['action_type'] == ActionType.ACCEPT
        assert action['price'] == 95.0
        
        # Test with no buyer offer - should make high offer
        self.state.last_buyer_offer = None
        action = seller._aggressive_strategy(self.state, [])
        assert action['action_type'] == ActionType.OFFER
        expected_aggressive_price = 100.0 * 0.95  # 95.0
        assert action['price'] == expected_aggressive_price
    
    def test_deceptive_strategy(self):
        """Test deceptive strategy"""
        buyer = RuleBasedAgent("buyer", Personality.DECEPTIVE)
        
        # Test in later rounds - should sometimes accept reasonable offers
        self.state.current_round = 6
        self.state.last_seller_offer = 110.0  # Within budget
        action = buyer._deceptive_strategy(self.state, [])
        assert action['action_type'] == ActionType.ACCEPT
        assert action['price'] == 110.0
        
        # Test with no seller offer - should make inconsistent offer
        self.state.last_seller_offer = None
        action = buyer._deceptive_strategy(self.state, [])
        assert action['action_type'] == ActionType.OFFER
        # Price should be between 66.0 and 78.0 (60*1.1 to 60*1.3)
        assert 66.0 <= action['price'] <= 78.0
    
    def test_rational_strategy(self):
        """Test rational strategy"""
        buyer = RuleBasedAgent("buyer", Personality.RATIONAL)
        
        # Test with acceptable seller offer
        self.state.last_seller_offer = 110.0  # Within acceptable range
        action = buyer._rational_strategy(self.state, [])
        assert action['action_type'] == ActionType.ACCEPT
        assert action['price'] == 110.0
        
        # Test with no seller offer - should make optimal offer
        self.state.last_seller_offer = None
        self.state.current_round = 5  # Middle rounds
        action = buyer._rational_strategy(self.state, [])
        assert action['action_type'] == ActionType.OFFER
        expected_optimal_price = 60.0 * 1.25  # 75.0
        assert action['price'] == expected_optimal_price
    
    def test_choose_action(self):
        """Test choose_action method"""
        buyer = RuleBasedAgent("buyer", Personality.COOPERATIVE)
        
        # Should call the appropriate strategy based on personality
        action = buyer.choose_action(self.state, self.env)
        assert isinstance(action, dict)
        assert 'action_type' in action
        assert 'price' in action

class TestQNetwork:
    """Test cases for Q-network"""
    
    def test_network_initialization(self):
        """Test Q-network initialization"""
        network = QNetwork(input_size=5, output_size=10)
        
        assert isinstance(network.fc1, torch.nn.Linear)
        assert isinstance(network.fc2, torch.nn.Linear)
        assert isinstance(network.fc3, torch.nn.Linear)
        assert network.fc1.in_features == 5
        assert network.fc3.out_features == 10
    
    def test_forward_pass(self):
        """Test forward pass through network"""
        network = QNetwork(input_size=5, output_size=10)
        x = torch.randn(1, 5)
        
        output = network(x)
        
        assert output.shape == (1, 10)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

class TestRLAgent:
    """Test cases for RL agents"""
    
    def setup_method(self):
        """Set up test environment and agent"""
        self.env = NegotiationEnvironment(
            max_rounds=10,
            item_value=100.0,
            buyer_budget=120.0,
            seller_cost=60.0
        )
        self.agent = RLAgent("buyer", self.env)
        self.state = self.env.reset()
    
    def test_agent_initialization(self):
        """Test RL agent initialization"""
        assert self.agent.agent_type == "buyer"
        assert self.agent.epsilon == 1.0
        assert self.agent.epsilon_min == 0.01
        assert self.agent.epsilon_decay == 0.995
        assert self.agent.gamma == 0.95
        assert len(self.agent.memory) == 0
        assert self.agent.memory_size == 10000
        assert self.agent.batch_size == 32
    
    def test_choose_action_exploration(self):
        """Test action choice during exploration"""
        # Set high epsilon for exploration
        self.agent.epsilon = 1.0
        
        action = self.agent.choose_action(self.state)
        
        assert isinstance(action, dict)
        assert 'action_type' in action
        assert 'price' in action
    
    def test_choose_action_exploitation(self):
        """Test action choice during exploitation"""
        # Set low epsilon for exploitation
        self.agent.epsilon = 0.0
        
        action = self.agent.choose_action(self.state)
        
        assert isinstance(action, dict)
        assert 'action_type' in action
        assert 'price' in action
    
    def test_remember(self):
        """Test experience storage"""
        action = {'action_type': ActionType.OFFER, 'price': 80.0}
        reward = 5.0
        next_state = self.env.reset()
        done = False
        
        initial_memory_size = len(self.agent.memory)
        self.agent.remember(self.state, action, reward, next_state, done)
        
        assert len(self.agent.memory) == initial_memory_size + 1
    
    def test_memory_size_limit(self):
        """Test that memory doesn't exceed size limit"""
        # Fill memory beyond limit
        for i in range(self.agent.memory_size + 10):
            action = {'action_type': ActionType.OFFER, 'price': 80.0 + i}
            reward = 5.0
            next_state = self.env.reset()
            done = False
            self.agent.remember(self.state, action, reward, next_state, done)
        
        assert len(self.agent.memory) <= self.agent.memory_size
    
    def test_replay_with_insufficient_memory(self):
        """Test replay when memory is insufficient"""
        # Should not crash when memory is too small
        self.agent.replay()
        # No assertion needed - just checking it doesn't crash
    
    def test_replay_with_sufficient_memory(self):
        """Test replay with sufficient memory"""
        # Fill memory with enough experiences
        for i in range(self.agent.batch_size + 5):
            action = {'action_type': ActionType.OFFER, 'price': 80.0 + i}
            reward = 5.0
            next_state = self.env.reset()
            done = False
            self.agent.remember(self.state, action, reward, next_state, done)
        
        # Should not crash during replay
        self.agent.replay()
        # No assertion needed - just checking it doesn't crash
    
    def test_epsilon_decay(self):
        """Test epsilon decay during replay"""
        initial_epsilon = self.agent.epsilon
        
        # Fill memory and replay
        for i in range(self.agent.batch_size + 5):
            action = {'action_type': ActionType.OFFER, 'price': 80.0 + i}
            reward = 5.0
            next_state = self.env.reset()
            done = False
            self.agent.remember(self.state, action, reward, next_state, done)
        
        self.agent.replay()
        
        # Epsilon should decay
        assert self.agent.epsilon < initial_epsilon
    
    def test_update_target_network(self):
        """Test target network update"""
        # Get initial weights
        initial_weights = self.agent.target_network.fc1.weight.clone()
        
        # Modify main network weights
        with torch.no_grad():
            self.agent.q_network.fc1.weight += 0.1
        
        # Update target network
        self.agent.update_target_network()
        
        # Target network should now match main network
        assert torch.allclose(
            self.agent.target_network.fc1.weight,
            self.agent.q_network.fc1.weight
        )
    
    def test_save_and_load_model(self):
        """Test model saving and loading"""
        import tempfile
        import os
        
        # Save model
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp_file:
            self.agent.save_model(tmp_file.name)
            
            # Load model into new agent
            new_agent = RLAgent("buyer", self.env)
            new_agent.load_model(tmp_file.name)
            
            # Check that weights are the same
            for param1, param2 in zip(self.agent.q_network.parameters(), 
                                    new_agent.q_network.parameters()):
                assert torch.allclose(param1, param2)
            
            # Check that epsilon is the same
            assert self.agent.epsilon == new_agent.epsilon
        
        # Clean up
        os.unlink(tmp_file.name)

class TestPersonality:
    """Test cases for personality constants"""
    
    def test_personality_constants(self):
        """Test that personality constants are defined"""
        assert Personality.COOPERATIVE == "cooperative"
        assert Personality.AGGRESSIVE == "aggressive"
        assert Personality.DECEPTIVE == "deceptive"
        assert Personality.RATIONAL == "rational"
