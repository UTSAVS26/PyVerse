import pytest
import numpy as np
import os
import tempfile
from train import TrainingManager
from environment import NegotiationEnvironment
from agents import RLAgent, RuleBasedAgent, Personality

class TestTrainingManager:
    """Test cases for the training manager"""
    
    def setup_method(self):
        """Set up test environment and training manager"""
        self.env = NegotiationEnvironment(
            max_rounds=10,
            item_value=100.0,
            buyer_budget=120.0,
            seller_cost=60.0
        )
        self.buyer_agent = RLAgent("buyer", self.env)
        self.seller_agent = RLAgent("seller", self.env)
        
        # Use temporary directory for results
        self.temp_dir = tempfile.mkdtemp()
        self.original_results_dir = 'results'
        
        # Create temporary results directory
        os.makedirs(os.path.join(self.temp_dir, 'results'), exist_ok=True)
        os.makedirs(os.path.join(self.temp_dir, 'models'), exist_ok=True)
        
        self.trainer = TrainingManager(
            env=self.env,
            buyer_agent=self.buyer_agent,
            seller_agent=self.seller_agent,
            training_episodes=10,  # Small number for testing
            evaluation_episodes=5
        )
    
    def teardown_method(self):
        """Clean up after tests"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_trainer_initialization(self):
        """Test training manager initialization"""
        assert self.trainer.env == self.env
        assert self.trainer.buyer_agent == self.buyer_agent
        assert self.trainer.seller_agent == self.seller_agent
        assert self.trainer.training_episodes == 10
        assert self.trainer.evaluation_episodes == 5
        
        # Check training history structure
        assert 'buyer_rewards' in self.trainer.training_history
        assert 'seller_rewards' in self.trainer.training_history
        assert 'deal_success_rate' in self.trainer.training_history
        assert 'average_deal_price' in self.trainer.training_history
        assert 'average_rounds' in self.trainer.training_history
    
    def test_calculate_episode_metrics_no_deal(self):
        """Test metrics calculation when no deal is made"""
        state = self.env.reset()
        # Add some negotiation history without acceptance
        state.negotiation_history = [
            {
                'round': 1,
                'buyer_action': {'action_type': 'offer', 'price': 80.0},
                'seller_action': {'action_type': 'offer', 'price': 90.0}
            },
            {
                'round': 2,
                'buyer_action': {'action_type': 'offer', 'price': 85.0},
                'seller_action': {'action_type': 'offer', 'price': 88.0}
            }
        ]
        
        success_rate, avg_price, avg_rounds = self.trainer._calculate_episode_metrics(state)
        
        assert success_rate == 0.0  # No deal made
        assert avg_price == 0.0  # No final price
        assert avg_rounds == 2.0  # Two rounds of negotiation
    
    def test_calculate_episode_metrics_with_deal(self):
        """Test metrics calculation when deal is made"""
        state = self.env.reset()
        # Add negotiation history with acceptance
        state.negotiation_history = [
            {
                'round': 1,
                'buyer_action': {'action_type': 'offer', 'price': 80.0},
                'seller_action': {'action_type': 'offer', 'price': 90.0}
            },
            {
                'round': 2,
                'buyer_action': {'action_type': 'accept', 'price': 85.0},
                'seller_action': {'action_type': 'offer', 'price': 85.0}
            }
        ]
        
        success_rate, avg_price, avg_rounds = self.trainer._calculate_episode_metrics(state)
        
        assert success_rate == 1.0  # Deal made
        assert avg_price == 85.0  # Final price
        assert avg_rounds == 2.0  # Two rounds of negotiation
    
    def test_calculate_episode_metrics_empty_history(self):
        """Test metrics calculation with empty history"""
        state = self.env.reset()
        state.negotiation_history = []
        
        success_rate, avg_price, avg_rounds = self.trainer._calculate_episode_metrics(state)
        
        assert success_rate == 0.0
        assert avg_price == 0.0
        assert avg_rounds == 0.0
    
    def test_evaluate_matchup(self):
        """Test evaluation matchup functionality"""
        # Create rule-based agents for evaluation
        rule_buyer = RuleBasedAgent("buyer", Personality.RATIONAL)
        rule_seller = RuleBasedAgent("seller", Personality.RATIONAL)
        
        results = self.trainer._evaluate_matchup(
            self.buyer_agent, self.seller_agent, rule_buyer, rule_seller
        )
        
        # Check results structure
        assert 'rl_buyer_rewards' in results
        assert 'rl_seller_rewards' in results
        assert 'rule_buyer_rewards' in results
        assert 'rule_seller_rewards' in results
        assert 'rl_buyer_avg' in results
        assert 'rl_seller_avg' in results
        assert 'rule_buyer_avg' in results
        assert 'rule_seller_avg' in results
        
        # Check that rewards are lists
        assert isinstance(results['rl_buyer_rewards'], list)
        assert isinstance(results['rl_seller_rewards'], list)
        assert isinstance(results['rule_buyer_rewards'], list)
        assert isinstance(results['rule_seller_rewards'], list)
        
        # Check that averages are calculated
        assert isinstance(results['rl_buyer_avg'], float)
        assert isinstance(results['rl_seller_avg'], float)
        assert isinstance(results['rule_buyer_avg'], float)
        assert isinstance(results['rule_seller_avg'], float)
    
    def test_evaluate(self):
        """Test evaluation functionality"""
        results = self.trainer.evaluate()
        
        # Check results structure
        assert 'rl_vs_rule' in results
        assert 'rl_vs_aggressive' in results
        assert 'rl_vs_cooperative' in results
        
        # Check that each matchup has the expected structure
        for matchup in results.values():
            assert 'rl_buyer_rewards' in matchup
            assert 'rl_seller_rewards' in matchup
            assert 'rule_buyer_rewards' in matchup
            assert 'rule_seller_rewards' in matchup
    
    def test_save_training_results(self):
        """Test saving training results"""
        # Add some dummy training history
        self.trainer.training_history['buyer_rewards'] = [1.0, 2.0, 3.0]
        self.trainer.training_history['seller_rewards'] = [2.0, 3.0, 4.0]
        self.trainer.training_history['deal_success_rate'] = [0.5, 0.7, 0.8]
        self.trainer.training_history['average_deal_price'] = [85.0, 87.0, 89.0]
        self.trainer.training_history['average_rounds'] = [3.0, 2.5, 2.0]
        
        # Create dummy evaluation results
        evaluation_results = {
            'rl_vs_rule': {
                'rl_buyer_avg': 2.0,
                'rl_seller_avg': 3.0,
                'rule_buyer_avg': 1.5,
                'rule_seller_avg': 2.5
            }
        }
        
        # Test saving (should not crash)
        self.trainer.save_training_results(evaluation_results)
        # No assertion needed - just checking it doesn't crash
    
    def test_create_training_plots(self):
        """Test training plots creation"""
        # Add some dummy training history
        self.trainer.training_history['buyer_rewards'] = [1.0, 2.0, 3.0]
        self.trainer.training_history['seller_rewards'] = [2.0, 3.0, 4.0]
        self.trainer.training_history['deal_success_rate'] = [0.5, 0.7, 0.8]
        self.trainer.training_history['average_deal_price'] = [85.0, 87.0, 89.0]
        self.trainer.training_history['average_rounds'] = [3.0, 2.5, 2.0]
        
        # Test plot creation (should not crash)
        self.trainer._create_training_plots("test_timestamp")
        # No assertion needed - just checking it doesn't crash

class TestTrainingIntegration:
    """Integration tests for training process"""
    
    def setup_method(self):
        """Set up test environment"""
        self.env = NegotiationEnvironment(
            max_rounds=5,  # Shorter for testing
            item_value=100.0,
            buyer_budget=120.0,
            seller_cost=60.0
        )
        self.buyer_agent = RLAgent("buyer", self.env)
        self.seller_agent = RLAgent("seller", self.env)
    
    def test_single_training_episode(self):
        """Test a single training episode"""
        trainer = TrainingManager(
            env=self.env,
            buyer_agent=self.buyer_agent,
            seller_agent=self.seller_agent,
            training_episodes=1,
            evaluation_episodes=1
        )
        
        # Run a single episode
        state = self.env.reset()
        episode_rewards = {'buyer': 0, 'seller': 0}
        
        while True:
            buyer_action = self.buyer_agent.choose_action(state)
            seller_action = self.seller_agent.choose_action(state)
            
            next_state, buyer_reward, seller_reward, done = self.env.step(
                state, buyer_action, seller_action
            )
            
            self.buyer_agent.remember(state, buyer_action, buyer_reward, next_state, done)
            self.seller_agent.remember(state, seller_action, seller_reward, next_state, done)
            
            episode_rewards['buyer'] += buyer_reward
            episode_rewards['seller'] += seller_reward
            
            state = next_state
            
            if done:
                break
        
        # Check that episode completed
        assert episode_rewards['buyer'] != 0 or episode_rewards['seller'] != 0
        assert len(state.negotiation_history) > 0
    
    def test_agent_memory_accumulation(self):
        """Test that agents accumulate memory during training"""
        initial_buyer_memory = len(self.buyer_agent.memory)
        initial_seller_memory = len(self.seller_agent.memory)
        
        # Run a few steps
        state = self.env.reset()
        for _ in range(3):
            buyer_action = self.buyer_agent.choose_action(state)
            seller_action = self.seller_agent.choose_action(state)
            
            next_state, buyer_reward, seller_reward, done = self.env.step(
                state, buyer_action, seller_action
            )
            
            self.buyer_agent.remember(state, buyer_action, buyer_reward, next_state, done)
            self.seller_agent.remember(state, seller_action, seller_reward, next_state, done)
            
            state = next_state
            if done:
                break
        
        # Check that memory increased
        assert len(self.buyer_agent.memory) > initial_buyer_memory
        assert len(self.seller_agent.memory) > initial_seller_memory
    
    def test_epsilon_decay(self):
        """Test that epsilon decays during training"""
        initial_epsilon = self.buyer_agent.epsilon
        
        # Fill memory and replay multiple times
        for _ in range(5):
            state = self.env.reset()
            buyer_action = self.buyer_agent.choose_action(state)
            seller_action = self.seller_agent.choose_action(state)
            
            next_state, buyer_reward, seller_reward, done = self.env.step(
                state, buyer_action, seller_action
            )
            
            self.buyer_agent.remember(state, buyer_action, buyer_reward, next_state, done)
            self.seller_agent.remember(state, seller_action, seller_reward, next_state, done)
            
            # Replay to trigger epsilon decay
            self.buyer_agent.replay()
            self.seller_agent.replay()
        
        # Check that epsilon decreased
        assert self.buyer_agent.epsilon < initial_epsilon
        assert self.seller_agent.epsilon < initial_epsilon
