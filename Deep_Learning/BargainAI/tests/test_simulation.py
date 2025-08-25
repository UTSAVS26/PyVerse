import pytest
import numpy as np
import os
import tempfile
from simulate import NegotiationSimulator
from environment import NegotiationEnvironment
from agents import RuleBasedAgent, Personality

class TestNegotiationSimulator:
    """Test cases for the negotiation simulator"""
    
    def setup_method(self):
        """Set up test environment and simulator"""
        self.env = NegotiationEnvironment(
            max_rounds=10,
            item_value=100.0,
            buyer_budget=120.0,
            seller_cost=60.0
        )
        self.simulator = NegotiationSimulator(self.env)
    
    def test_simulator_initialization(self):
        """Test simulator initialization"""
        assert self.simulator.env == self.env
        assert self.simulator.simulation_results == []
    
    def test_run_single_simulation(self):
        """Test single simulation run"""
        buyer_agent = RuleBasedAgent("buyer", Personality.COOPERATIVE)
        seller_agent = RuleBasedAgent("seller", Personality.AGGRESSIVE)
        
        result = self.simulator._run_single_simulation(buyer_agent, seller_agent)
        
        # Check result structure
        assert 'rounds' in result
        assert 'deal_made' in result
        assert 'final_price' in result
        assert 'buyer_reward' in result
        assert 'seller_reward' in result
        assert 'negotiation_history' in result
        
        # Check data types
        assert isinstance(result['rounds'], int)
        assert isinstance(result['deal_made'], bool)
        assert isinstance(result['final_price'], float)
        assert isinstance(result['buyer_reward'], float)
        assert isinstance(result['seller_reward'], float)
        assert isinstance(result['negotiation_history'], list)
        
        # Check logical constraints
        assert result['rounds'] > 0
        assert result['final_price'] >= 0
        assert len(result['negotiation_history']) > 0
    
    def test_run_simulation_multiple(self):
        """Test running multiple simulations"""
        buyer_agent = RuleBasedAgent("buyer", Personality.COOPERATIVE)
        seller_agent = RuleBasedAgent("seller", Personality.AGGRESSIVE)
        
        results = self.simulator.run_simulation(buyer_agent, seller_agent, num_simulations=5)
        
        # Check results structure
        assert 'simulations' in results
        assert 'summary' in results
        
        # Check simulations list
        assert len(results['simulations']) == 5
        for sim in results['simulations']:
            assert 'rounds' in sim
            assert 'deal_made' in sim
            assert 'final_price' in sim
            assert 'buyer_reward' in sim
            assert 'seller_reward' in sim
            assert 'negotiation_history' in sim
        
        # Check summary structure
        summary = results['summary']
        assert 'total_deals' in summary
        assert 'success_rate' in summary
        assert 'average_price' in summary
        assert 'average_rounds' in summary
        assert 'buyer_avg_reward' in summary
        assert 'seller_avg_reward' in summary
        
        # Check summary calculations
        assert summary['total_deals'] >= 0
        assert summary['total_deals'] <= 5
        assert 0.0 <= summary['success_rate'] <= 1.0
        assert summary['average_price'] >= 0
        assert summary['average_rounds'] > 0
    
    def test_create_dialogue_transcript(self):
        """Test dialogue transcript creation"""
        buyer_agent = RuleBasedAgent("buyer", Personality.COOPERATIVE)
        seller_agent = RuleBasedAgent("seller", Personality.AGGRESSIVE)
        
        # Run a simulation to get result
        result = self.simulator._run_single_simulation(buyer_agent, seller_agent)
        
        # Create transcript
        transcript = self.simulator.create_dialogue_transcript(result)
        
        # Check transcript structure
        assert isinstance(transcript, str)
        assert "NEGOTIATION DIALOGUE TRANSCRIPT" in transcript
        assert "Item Value: $100.0" in transcript
        assert "Buyer Budget: $120.0" in transcript
        assert "Seller Cost: $60.0" in transcript
        
        # Check that transcript contains round information
        if result['rounds'] > 0:
            assert "ROUND" in transcript
        
        # Check final outcome
        if result['deal_made']:
            assert "DEAL MADE" in transcript
            assert f"${result['final_price']:.2f}" in transcript
        else:
            assert "NO DEAL" in transcript
        
        # Check metrics
        assert f"Total Rounds: {result['rounds']}" in transcript
        assert f"Buyer Reward: {result['buyer_reward']:.2f}" in transcript
        assert f"Seller Reward: {result['seller_reward']:.2f}" in transcript
    
    def test_visualize_negotiation(self):
        """Test negotiation visualization"""
        buyer_agent = RuleBasedAgent("buyer", Personality.COOPERATIVE)
        seller_agent = RuleBasedAgent("seller", Personality.AGGRESSIVE)
        
        # Run a simulation to get result
        result = self.simulator._run_single_simulation(buyer_agent, seller_agent)
        
        # Test visualization (should not crash)
        self.simulator.visualize_negotiation(result)
        # No assertion needed - just checking it doesn't crash
    
    def test_visualize_negotiation_with_save(self):
        """Test negotiation visualization with save"""
        buyer_agent = RuleBasedAgent("buyer", Personality.COOPERATIVE)
        seller_agent = RuleBasedAgent("seller", Personality.AGGRESSIVE)
        
        # Run a simulation to get result
        result = self.simulator._run_single_simulation(buyer_agent, seller_agent)
        
        # Test visualization with save
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            self.simulator.visualize_negotiation(result, save_path=tmp_file.name)
            
            # Check that file was created
            assert os.path.exists(tmp_file.name)
            assert os.path.getsize(tmp_file.name) > 0
            
            # Clean up
            os.unlink(tmp_file.name)
    
    def test_compare_agents(self):
        """Test agent comparison functionality"""
        agent_configs = [
            (Personality.COOPERATIVE, Personality.AGGRESSIVE),
            (Personality.AGGRESSIVE, Personality.COOPERATIVE)
        ]
        
        results = self.simulator.compare_agents(agent_configs, num_simulations=3)
        
        # Check results structure
        assert len(results) == 2
        assert 'cooperative_vs_aggressive' in results
        assert 'aggressive_vs_cooperative' in results
        
        # Check each comparison result
        for config_name, result in results.items():
            assert 'simulations' in result
            assert 'summary' in result
            assert len(result['simulations']) == 3
    
    def test_create_comparison_visualization(self):
        """Test comparison visualization"""
        # Create dummy comparison results
        comparison_results = {
            'cooperative_vs_aggressive': {
                'summary': {
                    'success_rate': 0.8,
                    'average_price': 85.0,
                    'average_rounds': 3.0,
                    'buyer_avg_reward': 2.0,
                    'seller_avg_reward': 3.0
                }
            },
            'aggressive_vs_cooperative': {
                'summary': {
                    'success_rate': 0.6,
                    'average_price': 90.0,
                    'average_rounds': 4.0,
                    'buyer_avg_reward': 1.5,
                    'seller_avg_reward': 2.5
                }
            }
        }
        
        # Test visualization (should not crash)
        self.simulator.create_comparison_visualization(comparison_results)
        # No assertion needed - just checking it doesn't crash
    
    def test_create_comparison_visualization_with_save(self):
        """Test comparison visualization with save"""
        # Create dummy comparison results
        comparison_results = {
            'cooperative_vs_aggressive': {
                'summary': {
                    'success_rate': 0.8,
                    'average_price': 85.0,
                    'average_rounds': 3.0,
                    'buyer_avg_reward': 2.0,
                    'seller_avg_reward': 3.0
                }
            }
        }
        
        # Test visualization with save
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            self.simulator.create_comparison_visualization(comparison_results, save_path=tmp_file.name)
            
            # Check that file was created
            assert os.path.exists(tmp_file.name)
            assert os.path.getsize(tmp_file.name) > 0
            
            # Clean up
            os.unlink(tmp_file.name)

class TestSimulationIntegration:
    """Integration tests for simulation process"""
    
    def setup_method(self):
        """Set up test environment"""
        self.env = NegotiationEnvironment(
            max_rounds=5,  # Shorter for testing
            item_value=100.0,
            buyer_budget=120.0,
            seller_cost=60.0
        )
        self.simulator = NegotiationSimulator(self.env)
    
    def test_different_personality_combinations(self):
        """Test different personality combinations"""
        personalities = [
            Personality.COOPERATIVE,
            Personality.AGGRESSIVE,
            Personality.DECEPTIVE,
            Personality.RATIONAL
        ]
        
        for buyer_personality in personalities:
            for seller_personality in personalities:
                buyer_agent = RuleBasedAgent("buyer", buyer_personality)
                seller_agent = RuleBasedAgent("seller", seller_personality)
                
                # Run simulation
                result = self.simulator._run_single_simulation(buyer_agent, seller_agent)
                
                # Check basic constraints
                assert result['rounds'] > 0
                assert result['rounds'] <= 5  # Max rounds
                assert result['final_price'] >= 0
                assert len(result['negotiation_history']) > 0
    
    def test_simulation_consistency(self):
        """Test that simulations are consistent"""
        buyer_agent = RuleBasedAgent("buyer", Personality.RATIONAL)
        seller_agent = RuleBasedAgent("seller", Personality.RATIONAL)
        
        # Run multiple simulations
        results = []
        for _ in range(10):
            result = self.simulator._run_single_simulation(buyer_agent, seller_agent)
            results.append(result)
        
        # Check that all simulations completed
        for result in results:
            assert result['rounds'] > 0
            assert result['final_price'] >= 0
            assert len(result['negotiation_history']) > 0
        
        # Check that some deals were made (rational agents should be able to reach deals)
        deals_made = sum(1 for r in results if r['deal_made'])
        assert deals_made > 0  # At least some deals should be made
    
    def test_reward_distribution(self):
        """Test that rewards are distributed reasonably"""
        buyer_agent = RuleBasedAgent("buyer", Personality.COOPERATIVE)
        seller_agent = RuleBasedAgent("seller", Personality.COOPERATIVE)
        
        # Run multiple simulations
        buyer_rewards = []
        seller_rewards = []
        
        for _ in range(20):
            result = self.simulator._run_single_simulation(buyer_agent, seller_agent)
            buyer_rewards.append(result['buyer_reward'])
            seller_rewards.append(result['seller_reward'])
        
        # Check that rewards are reasonable
        assert np.mean(buyer_rewards) > -10  # Shouldn't be too negative
        assert np.mean(seller_rewards) > -10  # Shouldn't be too negative
        
        # Check that successful deals have positive rewards
        successful_results = [r for r in [self.simulator._run_single_simulation(buyer_agent, seller_agent) for _ in range(10)] if r['deal_made']]
        if successful_results:
            successful_buyer_rewards = [r['buyer_reward'] for r in successful_results]
            successful_seller_rewards = [r['seller_reward'] for r in successful_results]
            
            assert np.mean(successful_buyer_rewards) > 0
            assert np.mean(successful_seller_rewards) > 0
