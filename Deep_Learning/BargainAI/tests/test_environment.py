import pytest
import numpy as np
from environment import NegotiationEnvironment, NegotiationState, ActionType

class TestNegotiationEnvironment:
    """Test cases for the negotiation environment"""
    
    def setup_method(self):
        """Set up test environment before each test"""
        self.env = NegotiationEnvironment(
            max_rounds=10,
            item_value=100.0,
            buyer_budget=120.0,
            seller_cost=60.0
        )
    
    def test_environment_initialization(self):
        """Test environment initialization with correct parameters"""
        assert self.env.max_rounds == 10
        assert self.env.item_value == 100.0
        assert self.env.buyer_budget == 120.0
        assert self.env.seller_cost == 60.0
    
    def test_reset(self):
        """Test environment reset functionality"""
        state = self.env.reset()
        
        assert isinstance(state, NegotiationState)
        assert state.current_round == 0
        assert state.buyer_budget == 120.0
        assert state.seller_cost == 60.0
        assert state.item_value == 100.0
        assert state.last_buyer_offer is None
        assert state.last_seller_offer is None
        assert len(state.negotiation_history) == 0
    
    def test_acceptable_ranges(self):
        """Test that acceptable ranges are set correctly"""
        state = self.env.reset()
        
        # Buyer acceptable range should be between seller_cost*0.8 and buyer_budget*0.95
        expected_buyer_min = 60.0 * 0.8  # 48.0
        expected_buyer_max = 120.0 * 0.95  # 114.0
        assert state.buyer_acceptable_range[0] == expected_buyer_min
        assert state.buyer_acceptable_range[1] == expected_buyer_max
        
        # Seller acceptable range should be between seller_cost*1.1 and item_value*0.9
        expected_seller_min = 60.0 * 1.1  # 66.0
        expected_seller_max = 100.0 * 0.9  # 90.0
        assert state.seller_acceptable_range[0] == expected_seller_min
        assert state.seller_acceptable_range[1] == expected_seller_max
    
    def test_step_with_offers(self):
        """Test environment step with offers"""
        state = self.env.reset()
        
        buyer_action = {'action_type': ActionType.OFFER, 'price': 80.0}
        seller_action = {'action_type': ActionType.OFFER, 'price': 90.0}
        
        next_state, buyer_reward, seller_reward, done = self.env.step(state, buyer_action, seller_action)
        
        assert next_state.current_round == 1
        assert next_state.last_buyer_offer == 80.0
        assert next_state.last_seller_offer == 90.0
        assert len(next_state.negotiation_history) == 1
        assert not done  # Should not be done after first round
        assert buyer_reward == -0.1  # Small negative reward for continuing
        assert seller_reward == -0.1
    
    def test_step_with_acceptance(self):
        """Test environment step with acceptance"""
        state = self.env.reset()
        state.last_seller_offer = 85.0
        
        buyer_action = {'action_type': ActionType.ACCEPT, 'price': 85.0}
        seller_action = {'action_type': ActionType.OFFER, 'price': 90.0}
        
        next_state, buyer_reward, seller_reward, done = self.env.step(state, buyer_action, seller_action)
        
        assert done  # Should be done when deal is made
        assert buyer_reward > 0  # Buyer should get positive reward for good deal
        assert seller_reward > 0  # Seller should get positive reward for good deal
    
    def test_step_with_walk_away(self):
        """Test environment step with walk away"""
        state = self.env.reset()
        
        buyer_action = {'action_type': ActionType.WALK_AWAY, 'price': None}
        seller_action = {'action_type': ActionType.OFFER, 'price': 90.0}
        
        next_state, buyer_reward, seller_reward, done = self.env.step(state, buyer_action, seller_action)
        
        assert done  # Should be done when someone walks away
        assert buyer_reward == -10.0  # Buyer gets penalized for walking away
        assert seller_reward == -5.0  # Seller gets smaller penalty
    
    def test_max_rounds_reached(self):
        """Test that negotiation ends after max rounds"""
        state = self.env.reset()
        state.current_round = 10  # Set to max rounds
        
        buyer_action = {'action_type': ActionType.OFFER, 'price': 80.0}
        seller_action = {'action_type': ActionType.OFFER, 'price': 90.0}
        
        next_state, buyer_reward, seller_reward, done = self.env.step(state, buyer_action, seller_action)
        
        assert done  # Should be done after max rounds
        assert buyer_reward == -5.0  # Both get penalized for timeout
        assert seller_reward == -5.0
    
    def test_calculate_deal_rewards(self):
        """Test reward calculation for deals"""
        # Test fair deal
        buyer_reward, seller_reward = self.env._calculate_deal_rewards(85.0)
        assert buyer_reward > 0  # Buyer saves money
        assert seller_reward > 0  # Seller makes profit
        
        # Test buyer overpaying
        buyer_reward, seller_reward = self.env._calculate_deal_rewards(130.0)
        assert buyer_reward == -20.0  # Buyer gets penalized for overpaying
        assert seller_reward > 0  # Seller still gets positive reward
        
        # Test seller selling at loss
        buyer_reward, seller_reward = self.env._calculate_deal_rewards(50.0)
        assert buyer_reward > 0  # Buyer gets positive reward
        assert seller_reward == -20.0  # Seller gets penalized for selling at loss
    
    def test_get_valid_actions_buyer(self):
        """Test valid actions for buyer"""
        state = self.env.reset()
        actions = self.env.get_valid_actions("buyer", state)
        
        # Should have offer actions for different prices
        offer_actions = [a for a in actions if a['action_type'] == ActionType.OFFER]
        assert len(offer_actions) > 0
        
        # Should have walk away action
        walk_away_actions = [a for a in actions if a['action_type'] == ActionType.WALK_AWAY]
        assert len(walk_away_actions) == 1
        
        # Should not have accept/reject actions initially
        accept_actions = [a for a in actions if a['action_type'] == ActionType.ACCEPT]
        assert len(accept_actions) == 0
    
    def test_get_valid_actions_buyer_with_offer(self):
        """Test valid actions for buyer when seller has made an offer"""
        state = self.env.reset()
        state.last_seller_offer = 85.0
        
        actions = self.env.get_valid_actions("buyer", state)
        
        # Should have accept and reject actions
        accept_actions = [a for a in actions if a['action_type'] == ActionType.ACCEPT]
        reject_actions = [a for a in actions if a['action_type'] == ActionType.REJECT]
        assert len(accept_actions) == 1
        assert len(reject_actions) == 1
        assert accept_actions[0]['price'] == 85.0
    
    def test_get_valid_actions_seller(self):
        """Test valid actions for seller"""
        state = self.env.reset()
        actions = self.env.get_valid_actions("seller", state)
        
        # Should have offer actions for different prices
        offer_actions = [a for a in actions if a['action_type'] == ActionType.OFFER]
        assert len(offer_actions) > 0
        
        # Should have walk away action
        walk_away_actions = [a for a in actions if a['action_type'] == ActionType.WALK_AWAY]
        assert len(walk_away_actions) == 1
    
    def test_get_state_features(self):
        """Test state feature extraction"""
        state = self.env.reset()
        state.last_buyer_offer = 80.0
        state.last_seller_offer = 90.0
        
        # Test buyer features
        buyer_features = self.env.get_state_features(state, "buyer")
        assert len(buyer_features) == 7  # Should have 7 features for buyer
        assert buyer_features[0] == 0.0  # Normalized round
        assert buyer_features[1] == 1.2  # Budget ratio
        assert buyer_features[2] == 0.6  # Cost ratio
        assert buyer_features[3] == 0.8  # Last buyer offer ratio
        assert buyer_features[4] == 0.9  # Last seller offer ratio
        
        # Test seller features
        seller_features = self.env.get_state_features(state, "seller")
        assert len(seller_features) == 7  # Should have 7 features for seller
        assert seller_features[0] == 0.0  # Normalized round
        assert seller_features[1] == 1.2  # Budget ratio
        assert seller_features[2] == 0.6  # Cost ratio
        assert seller_features[3] == 0.8  # Last buyer offer ratio
        assert seller_features[4] == 0.9  # Last seller offer ratio

class TestNegotiationState:
    """Test cases for the negotiation state"""
    
    def test_state_initialization(self):
        """Test state initialization"""
        state = NegotiationState(
            current_round=5,
            buyer_budget=120.0,
            seller_cost=60.0,
            item_value=100.0
        )
        
        assert state.current_round == 5
        assert state.buyer_budget == 120.0
        assert state.seller_cost == 60.0
        assert state.item_value == 100.0
        assert state.last_buyer_offer is None
        assert state.last_seller_offer is None
        assert state.negotiation_history == []
    
    def test_state_with_history(self):
        """Test state with negotiation history"""
        history = [{'round': 1, 'buyer_action': {}, 'seller_action': {}}]
        state = NegotiationState(
            current_round=1,
            buyer_budget=120.0,
            seller_cost=60.0,
            item_value=100.0,
            negotiation_history=history
        )
        
        assert len(state.negotiation_history) == 1
        assert state.negotiation_history[0]['round'] == 1
