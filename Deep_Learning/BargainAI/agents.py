import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional
from environment import NegotiationState, ActionType, NegotiationEnvironment
import random

class Personality:
    """Defines different negotiation personalities"""
    COOPERATIVE = "cooperative"
    AGGRESSIVE = "aggressive"
    DECEPTIVE = "deceptive"
    RATIONAL = "rational"

class RuleBasedAgent:
    """Rule-based negotiation agent with different personalities"""
    
    def __init__(self, agent_type: str, personality: str = Personality.RATIONAL):
        self.agent_type = agent_type  # "buyer" or "seller"
        self.personality = personality
        
    def choose_action(self, state: NegotiationState, env: NegotiationEnvironment) -> Dict:
        """Choose action based on rules and personality"""
        valid_actions = env.get_valid_actions(self.agent_type, state)
        
        if self.personality == Personality.COOPERATIVE:
            return self._cooperative_strategy(state, valid_actions)
        elif self.personality == Personality.AGGRESSIVE:
            return self._aggressive_strategy(state, valid_actions)
        elif self.personality == Personality.DECEPTIVE:
            return self._deceptive_strategy(state, valid_actions)
        else:  # RATIONAL
            return self._rational_strategy(state, valid_actions)
    
    def _cooperative_strategy(self, state: NegotiationState, valid_actions: List[Dict]) -> Dict:
        """Cooperative strategy: aims for fair deals"""
        if self.agent_type == "buyer":
            # Accept if seller's offer is reasonable
            if state.last_seller_offer and state.last_seller_offer <= state.buyer_budget * 0.9:
                return {'action_type': ActionType.ACCEPT, 'price': state.last_seller_offer}
            
            # Make fair offers
            fair_price = (state.seller_cost + state.buyer_budget) / 2
            return {'action_type': ActionType.OFFER, 'price': fair_price}
        else:
            # Accept if buyer's offer is reasonable
            if state.last_buyer_offer and state.last_buyer_offer >= state.seller_cost * 1.2:
                return {'action_type': ActionType.ACCEPT, 'price': state.last_buyer_offer}
            
            # Make fair offers
            fair_price = (state.seller_cost + state.item_value) / 2
            return {'action_type': ActionType.OFFER, 'price': fair_price}
    
    def _aggressive_strategy(self, state: NegotiationState, valid_actions: List[Dict]) -> Dict:
        """Aggressive strategy: pushes for better deals"""
        if self.agent_type == "buyer":
            # Only accept very good offers
            if state.last_seller_offer and state.last_seller_offer <= state.seller_cost * 1.1:
                return {'action_type': ActionType.ACCEPT, 'price': state.last_seller_offer}
            
            # Make low offers
            aggressive_price = state.seller_cost * 1.05
            return {'action_type': ActionType.OFFER, 'price': aggressive_price}
        else:
            # Only accept high offers
            if state.last_buyer_offer and state.last_buyer_offer >= state.item_value * 0.9:
                return {'action_type': ActionType.ACCEPT, 'price': state.last_buyer_offer}
            
            # Make high offers
            aggressive_price = state.item_value * 0.95
            return {'action_type': ActionType.OFFER, 'price': aggressive_price}
    
    def _deceptive_strategy(self, state: NegotiationState, valid_actions: List[Dict]) -> Dict:
        """Deceptive strategy: bluffs and misleads"""
        if self.agent_type == "buyer":
            # Sometimes bluff by walking away
            if random.random() < 0.1 and state.current_round > 3:
                return {'action_type': ActionType.WALK_AWAY, 'price': None}
            
            # Pretend to be desperate in later rounds
            if state.current_round > 5:
                if state.last_seller_offer and state.last_seller_offer <= state.buyer_budget * 0.95:
                    return {'action_type': ActionType.ACCEPT, 'price': state.last_seller_offer}
            
            # Make inconsistent offers
            deceptive_price = state.seller_cost * (1.1 + random.random() * 0.3)
            return {'action_type': ActionType.OFFER, 'price': deceptive_price}
        else:
            # Sometimes bluff by walking away
            if random.random() < 0.1 and state.current_round > 3:
                return {'action_type': ActionType.WALK_AWAY, 'price': None}
            
            # Pretend to be desperate in later rounds
            if state.current_round > 5:
                if state.last_buyer_offer and state.last_buyer_offer >= state.seller_cost * 1.3:
                    return {'action_type': ActionType.ACCEPT, 'price': state.last_buyer_offer}
            
            # Make inconsistent offers
            deceptive_price = state.item_value * (0.7 + random.random() * 0.3)
            return {'action_type': ActionType.OFFER, 'price': deceptive_price}
    
    def _rational_strategy(self, state: NegotiationState, valid_actions: List[Dict]) -> Dict:
        """Rational strategy: optimizes based on utility"""
        if self.agent_type == "buyer":
            # Accept if seller's offer is within acceptable range
            if state.last_seller_offer and state.last_seller_offer <= state.buyer_acceptable_range[1]:
                return {'action_type': ActionType.ACCEPT, 'price': state.last_seller_offer}
            
            # Make optimal offer based on remaining rounds
            remaining_rounds = 10 - state.current_round
            if remaining_rounds <= 2:
                # Be more aggressive in final rounds
                optimal_price = state.seller_cost * 1.15
            else:
                optimal_price = state.seller_cost * 1.25
            
            return {'action_type': ActionType.OFFER, 'price': optimal_price}
        else:
            # Accept if buyer's offer is within acceptable range
            if state.last_buyer_offer and state.last_buyer_offer >= state.seller_acceptable_range[0]:
                return {'action_type': ActionType.ACCEPT, 'price': state.last_buyer_offer}
            
            # Make optimal offer based on remaining rounds
            remaining_rounds = 10 - state.current_round
            if remaining_rounds <= 2:
                # Be more cooperative in final rounds
                optimal_price = state.item_value * 0.85
            else:
                optimal_price = state.item_value * 0.9
            
            return {'action_type': ActionType.OFFER, 'price': optimal_price}

class QNetwork(nn.Module):
    """Neural network for Q-learning"""
    
    def __init__(self, input_size: int, output_size: int):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

class RLAgent:
    """Reinforcement Learning agent using Q-learning with neural networks"""
    
    def __init__(self, agent_type: str, env: NegotiationEnvironment, learning_rate: float = 0.001):
        self.agent_type = agent_type
        self.env = env
        
        # Get state and action dimensions
        sample_state = env.reset()
        self.state_size = len(env.get_state_features(sample_state, agent_type))
        self.action_size = len(env.get_valid_actions(agent_type, sample_state))
        
        # Q-networks
        self.q_network = QNetwork(self.state_size, self.action_size)
        self.target_network = QNetwork(self.state_size, self.action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Training parameters
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.95  # Discount factor
        
        # Experience replay
        self.memory = []
        self.memory_size = 10000
        self.batch_size = 32
        
    def choose_action(self, state: NegotiationState) -> Dict:
        """Choose action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            # Random action
            valid_actions = self.env.get_valid_actions(self.agent_type, state)
            return random.choice(valid_actions)
        else:
            # Best action according to Q-network
            state_features = self.env.get_state_features(state, self.agent_type)
            state_tensor = torch.FloatTensor(state_features).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            action_idx = q_values.argmax().item()
            
            valid_actions = self.env.get_valid_actions(self.agent_type, state)
            if action_idx < len(valid_actions):
                return valid_actions[action_idx]
            else:
                return random.choice(valid_actions)
    
    def remember(self, state: NegotiationState, action: Dict, reward: float, 
                next_state: NegotiationState, done: bool):
        """Store experience in replay memory"""
        state_features = self.env.get_state_features(state, self.agent_type)
        next_state_features = self.env.get_state_features(next_state, self.agent_type)
        
        # Convert action to index
        valid_actions = self.env.get_valid_actions(self.agent_type, state)
        try:
            action_idx = valid_actions.index(action)
        except ValueError:
            action_idx = 0
        
        # Ensure action_idx is within bounds
        action_idx = min(action_idx, self.action_size - 1)
        
        self.memory.append((state_features, action_idx, reward, next_state_features, done))
        
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
    
    def replay(self):
        """Train the Q-network on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor(np.array([exp[0] for exp in batch]))
        actions = torch.LongTensor([exp[1] for exp in batch])
        rewards = torch.FloatTensor([exp[2] for exp in batch])
        next_states = torch.FloatTensor(np.array([exp[3] for exp in batch]))
        dones = torch.BoolTensor([exp[4] for exp in batch])
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        """Update target network with current network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        checkpoint = torch.load(filepath)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
