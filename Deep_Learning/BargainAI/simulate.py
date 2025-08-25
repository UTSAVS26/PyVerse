import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import os
import json
from datetime import datetime

from environment import NegotiationEnvironment, NegotiationState, ActionType
from agents import RLAgent, RuleBasedAgent, Personality

class NegotiationSimulator:
    """Simulates and visualizes negotiation scenarios"""
    
    def __init__(self, env: NegotiationEnvironment):
        self.env = env
        self.simulation_results = []
        
    def run_simulation(self, buyer_agent, seller_agent, num_simulations: int = 10) -> Dict:
        """Run multiple negotiation simulations"""
        results = {
            'simulations': [],
            'summary': {
                'total_deals': 0,
                'success_rate': 0.0,
                'average_price': 0.0,
                'average_rounds': 0.0,
                'buyer_avg_reward': 0.0,
                'seller_avg_reward': 0.0
            }
        }
        
        for sim in range(num_simulations):
            simulation_result = self._run_single_simulation(buyer_agent, seller_agent)
            results['simulations'].append(simulation_result)
            
            # Update summary
            if simulation_result['deal_made']:
                results['summary']['total_deals'] += 1
                results['summary']['average_price'] += simulation_result['final_price']
        
        # Calculate averages
        results['summary']['success_rate'] = results['summary']['total_deals'] / num_simulations
        if results['summary']['total_deals'] > 0:
            results['summary']['average_price'] /= results['summary']['total_deals']
        
        results['summary']['average_rounds'] = np.mean([s['rounds'] for s in results['simulations']])
        results['summary']['buyer_avg_reward'] = np.mean([s['buyer_reward'] for s in results['simulations']])
        results['summary']['seller_avg_reward'] = np.mean([s['seller_reward'] for s in results['simulations']])
        
        return results
    
    def _run_single_simulation(self, buyer_agent, seller_agent) -> Dict:
        """Run a single negotiation simulation"""
        state = self.env.reset()
        round_num = 0
        buyer_reward = 0
        seller_reward = 0
        
        while True:
            round_num += 1
            
            # Agents choose actions
            if hasattr(buyer_agent, 'choose_action'):
                # Check if it's a rule-based agent (needs env parameter)
                if hasattr(buyer_agent, 'personality'):
                    buyer_action = buyer_agent.choose_action(state, self.env)
                else:
                    buyer_action = buyer_agent.choose_action(state)
            else:
                buyer_action = buyer_agent.choose_action(state, self.env)
                
            if hasattr(seller_agent, 'choose_action'):
                # Check if it's a rule-based agent (needs env parameter)
                if hasattr(seller_agent, 'personality'):
                    seller_action = seller_agent.choose_action(state, self.env)
                else:
                    seller_action = seller_agent.choose_action(state)
            else:
                seller_action = seller_agent.choose_action(state, self.env)
            
            # Environment step
            next_state, b_reward, s_reward, done = self.env.step(state, buyer_action, seller_action)
            
            buyer_reward += b_reward
            seller_reward += s_reward
            state = next_state
            
            if done:
                break
        
        # Determine if deal was made and final price
        deal_made = False
        final_price = 0.0
        
        for round_data in state.negotiation_history:
            buyer_action = round_data['buyer_action']
            seller_action = round_data['seller_action']
            
            if (buyer_action.get('action_type') == ActionType.ACCEPT or 
                seller_action.get('action_type') == ActionType.ACCEPT):
                deal_made = True
                if buyer_action.get('action_type') == ActionType.ACCEPT:
                    final_price = seller_action.get('price', 0.0)
                else:
                    final_price = buyer_action.get('price', 0.0)
                break
        
        return {
            'rounds': round_num,
            'deal_made': deal_made,
            'final_price': final_price,
            'buyer_reward': buyer_reward,
            'seller_reward': seller_reward,
            'negotiation_history': state.negotiation_history
        }
    
    def create_dialogue_transcript(self, simulation_result: Dict) -> str:
        """Create a human-readable dialogue transcript"""
        transcript = []
        transcript.append("🤝 NEGOTIATION DIALOGUE TRANSCRIPT")
        transcript.append("=" * 50)
        transcript.append(f"Item Value: ${self.env.item_value}")
        transcript.append(f"Buyer Budget: ${self.env.buyer_budget}")
        transcript.append(f"Seller Cost: ${self.env.seller_cost}")
        transcript.append("=" * 50)
        
        for i, round_data in enumerate(simulation_result['negotiation_history'], 1):
            buyer_action = round_data['buyer_action']
            seller_action = round_data['seller_action']
            
            transcript.append(f"\n📋 ROUND {i}")
            transcript.append("-" * 20)
            
            # Buyer's action
            if buyer_action.get('action_type') == ActionType.OFFER:
                transcript.append(f"👤 Buyer: \"I offer ${buyer_action.get('price'):.2f}\"")
            elif buyer_action.get('action_type') == ActionType.ACCEPT:
                transcript.append(f"👤 Buyer: \"I accept your offer of ${buyer_action.get('price'):.2f}!\"")
            elif buyer_action.get('action_type') == ActionType.REJECT:
                transcript.append(f"👤 Buyer: \"I reject your offer.\"")
            elif buyer_action.get('action_type') == ActionType.WALK_AWAY:
                transcript.append(f"👤 Buyer: \"I'm walking away from this deal.\"")
            
            # Seller's action
            if seller_action.get('action_type') == ActionType.OFFER:
                transcript.append(f"🏪 Seller: \"I offer ${seller_action.get('price'):.2f}\"")
            elif seller_action.get('action_type') == ActionType.ACCEPT:
                transcript.append(f"🏪 Seller: \"I accept your offer of ${seller_action.get('price'):.2f}!\"")
            elif seller_action.get('action_type') == ActionType.REJECT:
                transcript.append(f"🏪 Seller: \"I reject your offer.\"")
            elif seller_action.get('action_type') == ActionType.WALK_AWAY:
                transcript.append(f"🏪 Seller: \"I'm walking away from this deal.\"")
        
        # Final outcome
        transcript.append("\n" + "=" * 50)
        if simulation_result['deal_made']:
            transcript.append(f"✅ DEAL MADE!")
            transcript.append(f"Final Price: ${simulation_result['final_price']:.2f}")
            transcript.append(f"Buyer Surplus: ${self.env.buyer_budget - simulation_result['final_price']:.2f}")
            transcript.append(f"Seller Profit: ${simulation_result['final_price'] - self.env.seller_cost:.2f}")
        else:
            transcript.append("❌ NO DEAL")
            transcript.append("Negotiation failed - no agreement reached")
        
        transcript.append(f"Total Rounds: {simulation_result['rounds']}")
        transcript.append(f"Buyer Reward: {simulation_result['buyer_reward']:.2f}")
        transcript.append(f"Seller Reward: {simulation_result['seller_reward']:.2f}")
        
        return "\n".join(transcript)
    
    def visualize_negotiation(self, simulation_result: Dict, save_path: str = None):
        """Create visualization of the negotiation"""
        history = simulation_result['negotiation_history']
        
        if not history:
            print("No negotiation history to visualize")
            return
        
        # Extract price offers over time
        rounds = []
        buyer_offers = []
        seller_offers = []
        
        for i, round_data in enumerate(history, 1):
            rounds.append(i)
            
            buyer_action = round_data['buyer_action']
            seller_action = round_data['seller_action']
            
            if buyer_action.get('action_type') == ActionType.OFFER:
                buyer_offers.append(buyer_action.get('price'))
            else:
                buyer_offers.append(None)
            
            if seller_action.get('action_type') == ActionType.OFFER:
                seller_offers.append(seller_action.get('price'))
            else:
                seller_offers.append(None)
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Price negotiation plot
        ax1.plot(rounds, buyer_offers, 'bo-', label='Buyer Offers', markersize=8)
        ax1.plot(rounds, seller_offers, 'ro-', label='Seller Offers', markersize=8)
        
        # Add reference lines
        ax1.axhline(y=self.env.buyer_budget, color='g', linestyle='--', alpha=0.7, label='Buyer Budget')
        ax1.axhline(y=self.env.seller_cost, color='orange', linestyle='--', alpha=0.7, label='Seller Cost')
        ax1.axhline(y=self.env.item_value, color='purple', linestyle='--', alpha=0.7, label='Item Value')
        
        if simulation_result['deal_made']:
            ax1.axhline(y=simulation_result['final_price'], color='black', linestyle='-', 
                       linewidth=2, label=f'Final Price: ${simulation_result["final_price"]:.2f}')
        
        ax1.set_xlabel('Negotiation Round')
        ax1.set_ylabel('Price ($)')
        ax1.set_title('Price Negotiation Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Action type plot
        buyer_actions = [round_data['buyer_action'].get('action_type').value for round_data in history]
        seller_actions = [round_data['seller_action'].get('action_type').value for round_data in history]
        
        action_types = list(set(buyer_actions + seller_actions))
        colors = plt.cm.Set3(np.linspace(0, 1, len(action_types)))
        
        ax2.bar([r - 0.2 for r in rounds], [action_types.index(a) for a in buyer_actions], 
                width=0.4, label='Buyer', color='blue', alpha=0.7)
        ax2.bar([r + 0.2 for r in rounds], [action_types.index(a) for a in seller_actions], 
                width=0.4, label='Seller', color='red', alpha=0.7)
        
        ax2.set_xlabel('Negotiation Round')
        ax2.set_ylabel('Action Type')
        ax2.set_title('Action Types Over Time')
        ax2.set_yticks(range(len(action_types)))
        ax2.set_yticklabels(action_types)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def compare_agents(self, agent_configs: List[Tuple], num_simulations: int = 50) -> Dict:
        """Compare different agent configurations"""
        comparison_results = {}
        
        for buyer_type, seller_type in agent_configs:
            # Create agents
            if buyer_type == 'RL':
                buyer_agent = RLAgent("buyer", self.env)
                if os.path.exists('models/buyer_model.pth'):
                    buyer_agent.load_model('models/buyer_model.pth')
            else:
                buyer_agent = RuleBasedAgent("buyer", buyer_type)
            
            if seller_type == 'RL':
                seller_agent = RLAgent("seller", self.env)
                if os.path.exists('models/seller_model.pth'):
                    seller_agent.load_model('models/seller_model.pth')
            else:
                seller_agent = RuleBasedAgent("seller", seller_type)
            
            # Run simulations
            results = self.run_simulation(buyer_agent, seller_agent, num_simulations)
            comparison_results[f"{buyer_type}_vs_{seller_type}"] = results
        
        return comparison_results
    
    def create_comparison_visualization(self, comparison_results: Dict, save_path: str = None):
        """Create comparison visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Extract data
        configs = list(comparison_results.keys())
        success_rates = [comp['summary']['success_rate'] for comp in comparison_results.values()]
        avg_prices = [comp['summary']['average_price'] for comp in comparison_results.values()]
        avg_rounds = [comp['summary']['average_rounds'] for comp in comparison_results.values()]
        buyer_rewards = [comp['summary']['buyer_avg_reward'] for comp in comparison_results.values()]
        seller_rewards = [comp['summary']['seller_avg_reward'] for comp in comparison_results.values()]
        
        # Success rates
        axes[0, 0].bar(configs, success_rates, color='green', alpha=0.7)
        axes[0, 0].set_title('Deal Success Rates')
        axes[0, 0].set_ylabel('Success Rate')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Average prices
        axes[0, 1].bar(configs, avg_prices, color='orange', alpha=0.7)
        axes[0, 1].set_title('Average Deal Prices')
        axes[0, 1].set_ylabel('Price ($)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Average rounds
        axes[1, 0].bar(configs, avg_rounds, color='red', alpha=0.7)
        axes[1, 0].set_title('Average Rounds per Deal')
        axes[1, 0].set_ylabel('Rounds')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Rewards comparison
        x = np.arange(len(configs))
        width = 0.35
        axes[1, 1].bar(x - width/2, buyer_rewards, width, label='Buyer', alpha=0.7)
        axes[1, 1].bar(x + width/2, seller_rewards, width, label='Seller', alpha=0.7)
        axes[1, 1].set_title('Average Rewards')
        axes[1, 1].set_ylabel('Reward')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(configs, rotation=45)
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()

def main():
    """Main simulation function"""
    # Create environment
    env = NegotiationEnvironment(
        max_rounds=10,
        item_value=100.0,
        buyer_budget=120.0,
        seller_cost=60.0
    )
    
    # Create simulator
    simulator = NegotiationSimulator(env)
    
    # Create agents for demonstration
    cooperative_buyer = RuleBasedAgent("buyer", Personality.COOPERATIVE)
    aggressive_seller = RuleBasedAgent("seller", Personality.AGGRESSIVE)
    
    # Run single simulation with dialogue
    print("Running single negotiation simulation...")
    single_result = simulator._run_single_simulation(cooperative_buyer, aggressive_seller)
    
    # Print dialogue transcript
    transcript = simulator.create_dialogue_transcript(single_result)
    print(transcript)
    
    # Create visualization
    simulator.visualize_negotiation(single_result, 'results/single_negotiation.png')
    
    # Run comparison simulations
    print("\nRunning agent comparison...")
    agent_configs = [
        (Personality.COOPERATIVE, Personality.AGGRESSIVE),
        (Personality.AGGRESSIVE, Personality.COOPERATIVE),
        (Personality.RATIONAL, Personality.DECEPTIVE),
        (Personality.DECEPTIVE, Personality.RATIONAL)
    ]
    
    comparison_results = simulator.compare_agents(agent_configs, num_simulations=20)
    
    # Create comparison visualization
    simulator.create_comparison_visualization(comparison_results, 'results/agent_comparison.png')
    
    # Print comparison summary
    print("\n" + "=" * 60)
    print("AGENT COMPARISON SUMMARY")
    print("=" * 60)
    
    for config, results in comparison_results.items():
        summary = results['summary']
        print(f"\n{config}:")
        print(f"  Success Rate: {summary['success_rate']:.2%}")
        print(f"  Average Price: ${summary['average_price']:.2f}")
        print(f"  Average Rounds: {summary['average_rounds']:.1f}")
        print(f"  Buyer Reward: {summary['buyer_avg_reward']:.2f}")
        print(f"  Seller Reward: {summary['seller_avg_reward']:.2f}")
    
    print(f"\nResults saved in 'results/' directory")

if __name__ == "__main__":
    main()
