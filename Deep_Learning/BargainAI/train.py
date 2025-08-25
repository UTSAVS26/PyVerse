import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import os
import json
from datetime import datetime

from environment import NegotiationEnvironment, NegotiationState
from agents import RLAgent, RuleBasedAgent, Personality

class TrainingManager:
    """Manages the training process for negotiation agents"""
    
    def __init__(self, 
                 env: NegotiationEnvironment,
                 buyer_agent: RLAgent,
                 seller_agent: RLAgent,
                 training_episodes: int = 1000,
                 evaluation_episodes: int = 100):
        self.env = env
        self.buyer_agent = buyer_agent
        self.seller_agent = seller_agent
        self.training_episodes = training_episodes
        self.evaluation_episodes = evaluation_episodes
        
        # Training metrics
        self.training_history = {
            'buyer_rewards': [],
            'seller_rewards': [],
            'deal_success_rate': [],
            'average_deal_price': [],
            'average_rounds': []
        }
        
        # Create results directory
        os.makedirs('results', exist_ok=True)
        os.makedirs('models', exist_ok=True)
    
    def train(self) -> Dict:
        """Main training loop"""
        print(f"Starting training for {self.training_episodes} episodes...")
        
        for episode in range(self.training_episodes):
            # Reset environment
            state = self.env.reset()
            episode_rewards = {'buyer': 0, 'seller': 0}
            episode_rounds = 0
            
            while True:
                # Agents choose actions
                buyer_action = self.buyer_agent.choose_action(state)
                seller_action = self.seller_agent.choose_action(state)
                
                # Environment step
                next_state, buyer_reward, seller_reward, done = self.env.step(
                    state, buyer_action, seller_action
                )
                
                # Store experiences
                self.buyer_agent.remember(state, buyer_action, buyer_reward, next_state, done)
                self.seller_agent.remember(state, seller_action, seller_reward, next_state, done)
                
                # Update episode metrics
                episode_rewards['buyer'] += buyer_reward
                episode_rewards['seller'] += seller_reward
                episode_rounds += 1
                
                # Train agents
                self.buyer_agent.replay()
                self.seller_agent.replay()
                
                state = next_state
                
                if done:
                    break
            
            # Update target networks periodically
            if episode % 100 == 0:
                self.buyer_agent.update_target_network()
                self.seller_agent.update_target_network()
            
            # Record metrics
            self.training_history['buyer_rewards'].append(episode_rewards['buyer'])
            self.training_history['seller_rewards'].append(episode_rewards['seller'])
            
            # Calculate additional metrics
            success_rate, avg_price, avg_rounds = self._calculate_episode_metrics(state)
            self.training_history['deal_success_rate'].append(success_rate)
            self.training_history['average_deal_price'].append(avg_price)
            self.training_history['average_rounds'].append(avg_rounds)
            
            # Print progress
            if episode % 100 == 0:
                print(f"Episode {episode}/{self.training_episodes}")
                print(f"  Buyer Reward: {episode_rewards['buyer']:.2f}")
                print(f"  Seller Reward: {episode_rewards['seller']:.2f}")
                print(f"  Success Rate: {success_rate:.2f}")
                print(f"  Avg Price: {avg_price:.2f}")
                print(f"  Avg Rounds: {avg_rounds:.2f}")
                print(f"  Buyer Epsilon: {self.buyer_agent.epsilon:.3f}")
                print(f"  Seller Epsilon: {self.seller_agent.epsilon:.3f}")
                print("-" * 50)
        
        # Save trained models
        self.buyer_agent.save_model('models/buyer_model.pth')
        self.seller_agent.save_model('models/seller_model.pth')
        
        # Evaluate and save results
        evaluation_results = self.evaluate()
        self.save_training_results(evaluation_results)
        
        return evaluation_results
    
    def _calculate_episode_metrics(self, final_state: NegotiationState) -> Tuple[float, float, float]:
        """Calculate metrics for the episode"""
        history = final_state.negotiation_history
        
        if not history:
            return 0.0, 0.0, 0.0
        
        # Check if deal was made
        deal_made = False
        final_price = 0.0
        
        for round_data in history:
            buyer_action = round_data['buyer_action']
            seller_action = round_data['seller_action']
            
            if (buyer_action.get('action_type') == 'accept' or 
                seller_action.get('action_type') == 'accept'):
                deal_made = True
                if buyer_action.get('action_type') == 'accept':
                    final_price = seller_action.get('price', 0.0)
                else:
                    final_price = buyer_action.get('price', 0.0)
                break
        
        success_rate = 1.0 if deal_made else 0.0
        avg_price = final_price if deal_made else 0.0
        avg_rounds = len(history)
        
        return success_rate, avg_price, avg_rounds
    
    def evaluate(self) -> Dict:
        """Evaluate trained agents against rule-based agents"""
        print("Evaluating trained agents...")
        
        # Create rule-based agents for evaluation
        rule_buyer = RuleBasedAgent("buyer", Personality.RATIONAL)
        rule_seller = RuleBasedAgent("seller", Personality.RATIONAL)
        
        evaluation_results = {
            'rl_vs_rule': self._evaluate_matchup(self.buyer_agent, self.seller_agent, rule_buyer, rule_seller),
            'rl_vs_aggressive': self._evaluate_matchup(self.buyer_agent, self.seller_agent, 
                                                     RuleBasedAgent("buyer", Personality.AGGRESSIVE),
                                                     RuleBasedAgent("seller", Personality.AGGRESSIVE)),
            'rl_vs_cooperative': self._evaluate_matchup(self.buyer_agent, self.seller_agent,
                                                      RuleBasedAgent("buyer", Personality.COOPERATIVE),
                                                      RuleBasedAgent("seller", Personality.COOPERATIVE))
        }
        
        return evaluation_results
    
    def _evaluate_matchup(self, rl_buyer: RLAgent, rl_seller: RLAgent, 
                         rule_buyer: RuleBasedAgent, rule_seller: RuleBasedAgent) -> Dict:
        """Evaluate RL agents against rule-based agents"""
        
        # Set RL agents to evaluation mode (no exploration)
        rl_buyer.epsilon = 0.0
        rl_seller.epsilon = 0.0
        
        results = {
            'rl_buyer_rewards': [],
            'rl_seller_rewards': [],
            'rule_buyer_rewards': [],
            'rule_seller_rewards': [],
            'rl_success_rate': 0.0,
            'rule_success_rate': 0.0,
            'rl_avg_price': 0.0,
            'rule_avg_price': 0.0
        }
        
        # Test RL agents as buyer/seller
        for episode in range(self.evaluation_episodes):
            state = self.env.reset()
            episode_rewards = {'rl_buyer': 0, 'rule_seller': 0}
            
            while True:
                buyer_action = rl_buyer.choose_action(state)
                seller_action = rule_seller.choose_action(state, self.env)
                
                next_state, buyer_reward, seller_reward, done = self.env.step(
                    state, buyer_action, seller_action
                )
                
                episode_rewards['rl_buyer'] += buyer_reward
                episode_rewards['rule_seller'] += seller_reward
                
                state = next_state
                if done:
                    break
            
            results['rl_buyer_rewards'].append(episode_rewards['rl_buyer'])
            results['rule_seller_rewards'].append(episode_rewards['rule_seller'])
        
        # Test rule agents as buyer/seller
        for episode in range(self.evaluation_episodes):
            state = self.env.reset()
            episode_rewards = {'rule_buyer': 0, 'rl_seller': 0}
            
            while True:
                buyer_action = rule_buyer.choose_action(state, self.env)
                seller_action = rl_seller.choose_action(state)
                
                next_state, buyer_reward, seller_reward, done = self.env.step(
                    state, buyer_action, seller_action
                )
                
                episode_rewards['rule_buyer'] += buyer_reward
                episode_rewards['rl_seller'] += seller_reward
                
                state = next_state
                if done:
                    break
            
            results['rule_buyer_rewards'].append(episode_rewards['rule_buyer'])
            results['rl_seller_rewards'].append(episode_rewards['rl_seller'])
        
        # Calculate averages
        results['rl_buyer_avg'] = np.mean(results['rl_buyer_rewards'])
        results['rl_seller_avg'] = np.mean(results['rl_seller_rewards'])
        results['rule_buyer_avg'] = np.mean(results['rule_buyer_rewards'])
        results['rule_seller_avg'] = np.mean(results['rule_seller_rewards'])
        
        return results
    
    def save_training_results(self, evaluation_results: Dict):
        """Save training results and plots"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save training history
        with open(f'results/training_history_{timestamp}.json', 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        # Save evaluation results
        with open(f'results/evaluation_results_{timestamp}.json', 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        # Create training plots
        self._create_training_plots(timestamp)
    
    def _create_training_plots(self, timestamp: str):
        """Create and save training visualization plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot rewards
        axes[0, 0].plot(self.training_history['buyer_rewards'], label='Buyer', alpha=0.7)
        axes[0, 0].plot(self.training_history['seller_rewards'], label='Seller', alpha=0.7)
        axes[0, 0].set_title('Training Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot success rate
        axes[0, 1].plot(self.training_history['deal_success_rate'], color='green', alpha=0.7)
        axes[0, 1].set_title('Deal Success Rate')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Success Rate')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot average price
        axes[1, 0].plot(self.training_history['average_deal_price'], color='orange', alpha=0.7)
        axes[1, 0].set_title('Average Deal Price')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Price')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot average rounds
        axes[1, 1].plot(self.training_history['average_rounds'], color='red', alpha=0.7)
        axes[1, 1].set_title('Average Rounds per Deal')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Rounds')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'results/training_plots_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main training function"""
    # Create environment
    env = NegotiationEnvironment(
        max_rounds=10,
        item_value=100.0,
        buyer_budget=120.0,
        seller_cost=60.0
    )
    
    # Create RL agents
    buyer_agent = RLAgent("buyer", env)
    seller_agent = RLAgent("seller", env)
    
    # Create training manager
    trainer = TrainingManager(
        env=env,
        buyer_agent=buyer_agent,
        seller_agent=seller_agent,
        training_episodes=50,  # Shorter for testing
        evaluation_episodes=10
    )
    
    # Start training
    results = trainer.train()
    
    print("Training completed!")
    print("Results saved in 'results/' directory")
    print("Models saved in 'models/' directory")

if __name__ == "__main__":
    main()
