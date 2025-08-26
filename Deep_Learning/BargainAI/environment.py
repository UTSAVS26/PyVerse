import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class ActionType(Enum):
    OFFER = "offer"
    ACCEPT = "accept"
    REJECT = "reject"
    COUNTER_OFFER = "counter_offer"
    WALK_AWAY = "walk_away"

@dataclass
class NegotiationState:
    """Represents the current state of a negotiation"""
    current_round: int
    buyer_budget: float
    seller_cost: float
    item_value: float
    last_buyer_offer: Optional[float] = None
    last_seller_offer: Optional[float] = None
    buyer_acceptable_range: Tuple[float, float] = None
    seller_acceptable_range: Tuple[float, float] = None
    negotiation_history: List[Dict] = None
    
    def __post_init__(self):
        if self.negotiation_history is None:
            self.negotiation_history = []
        if self.buyer_acceptable_range is None:
            self.buyer_acceptable_range = (self.seller_cost * 0.8, self.buyer_budget * 0.95)
        if self.seller_acceptable_range is None:
            self.seller_acceptable_range = (self.seller_cost * 1.1, self.item_value * 0.9)

class NegotiationEnvironment:
    """Main negotiation environment that manages the game rules and state"""
    
    def __init__(self, 
                 max_rounds: int = 10,
                 item_value: float = 100.0,
                 buyer_budget: float = 120.0,
                 seller_cost: float = 60.0):
        self.max_rounds = max_rounds
        self.item_value = item_value
        self.buyer_budget = buyer_budget
        self.seller_cost = seller_cost
        
    def reset(self) -> NegotiationState:
        """Reset the environment to initial state"""
        return NegotiationState(
            current_round=0,
            buyer_budget=self.buyer_budget,
            seller_cost=self.seller_cost,
            item_value=self.item_value
        )
    
    def step(self, state: NegotiationState, buyer_action: Dict, seller_action: Dict) -> Tuple[NegotiationState, Dict, Dict, bool]:
        """
        Execute one step of the negotiation
        
        Args:
            state: Current negotiation state
            buyer_action: Action from buyer agent
            seller_action: Action from seller agent
            
        Returns:
            new_state, buyer_reward, seller_reward, done
        """
        new_state = NegotiationState(
            current_round=state.current_round + 1,
            buyer_budget=state.buyer_budget,
            seller_cost=state.seller_cost,
            item_value=state.item_value,
            buyer_acceptable_range=state.buyer_acceptable_range,
            seller_acceptable_range=state.seller_acceptable_range,
            negotiation_history=state.negotiation_history.copy()
        )
        
        # Process actions
        buyer_reward, seller_reward, done = self._process_actions(new_state, buyer_action, seller_action)
        
        return new_state, buyer_reward, seller_reward, done
    
    def _process_actions(self, state: NegotiationState, buyer_action: Dict, seller_action: Dict) -> Tuple[float, float, bool]:
        """Process the actions and determine rewards"""
        
        buyer_action_type = buyer_action.get('action_type')
        seller_action_type = seller_action.get('action_type')
        
        # Record actions in history
        state.negotiation_history.append({
            'round': state.current_round,
            'buyer_action': buyer_action,
            'seller_action': seller_action
        })
        
        # Handle walk away
        if buyer_action_type == ActionType.WALK_AWAY or seller_action_type == ActionType.WALK_AWAY:
            return -10.0, -5.0, True  # Both lose, but seller loses less
        
        # Handle acceptance
        if buyer_action_type == ActionType.ACCEPT or seller_action_type == ActionType.ACCEPT:
            if buyer_action_type == ActionType.ACCEPT:
                final_price = seller_action.get('price', state.last_seller_offer)
            else:
                final_price = buyer_action.get('price', state.last_buyer_offer)
            
            # Ensure final_price is not None
            if final_price is None:
                # Use a reasonable default price
                final_price = (self.seller_cost + self.buyer_budget) / 2
            
            buyer_reward, seller_reward = self._calculate_deal_rewards(final_price)
            return buyer_reward, seller_reward, True
        
        # Handle offers and counter-offers
        if buyer_action_type == ActionType.OFFER:
            state.last_buyer_offer = buyer_action.get('price')
        if seller_action_type == ActionType.OFFER:
            state.last_seller_offer = seller_action.get('price')
        
        # Check if max rounds reached
        if state.current_round >= self.max_rounds:
            return -5.0, -5.0, True  # Both lose due to time out
        
        # Small negative reward for continuing (encourages faster deals)
        return -0.1, -0.1, False
    
    def _calculate_deal_rewards(self, final_price: float) -> Tuple[float, float]:
        """Calculate rewards for both parties when a deal is made"""
        
        # Buyer reward: budget saved
        buyer_surplus = self.buyer_budget - final_price
        buyer_reward = buyer_surplus / self.buyer_budget * 10  # Normalize to reasonable scale
        
        # Seller reward: profit made
        seller_profit = final_price - self.seller_cost
        seller_reward = seller_profit / self.seller_cost * 10  # Normalize to reasonable scale
        
        # Penalize if price is outside acceptable ranges
        if final_price > self.buyer_budget:
            buyer_reward = -20.0  # Buyer overpaid
        if final_price < self.seller_cost:
            seller_reward = -20.0  # Seller sold at loss
        
        return buyer_reward, seller_reward
    
    def get_valid_actions(self, agent_type: str, state: NegotiationState) -> List[Dict]:
        """Get valid actions for an agent"""
        actions = []
        
        if agent_type == "buyer":
            # Buyer can offer, accept, reject, or walk away
            if state.last_seller_offer is not None:
                actions.append({
                    'action_type': ActionType.ACCEPT,
                    'price': state.last_seller_offer
                })
                actions.append({
                    'action_type': ActionType.REJECT,
                    'price': None
                })
            
            # Buyer can make offers
            for price in np.arange(self.seller_cost, self.buyer_budget + 1, 5):
                actions.append({
                    'action_type': ActionType.OFFER,
                    'price': price
                })
            
            actions.append({
                'action_type': ActionType.WALK_AWAY,
                'price': None
            })
            
        elif agent_type == "seller":
            # Seller can offer, accept, reject, or walk away
            if state.last_buyer_offer is not None:
                actions.append({
                    'action_type': ActionType.ACCEPT,
                    'price': state.last_buyer_offer
                })
                actions.append({
                    'action_type': ActionType.REJECT,
                    'price': None
                })
            
            # Seller can make offers
            for price in np.arange(self.seller_cost, self.item_value + 1, 5):
                actions.append({
                    'action_type': ActionType.OFFER,
                    'price': price
                })
            
            actions.append({
                'action_type': ActionType.WALK_AWAY,
                'price': None
            })
        
        return actions
    
    def get_state_features(self, state: NegotiationState, agent_type: str) -> np.ndarray:
        """Convert state to feature vector for RL agents"""
        features = [
            state.current_round / self.max_rounds,  # Normalized round
            state.buyer_budget / self.item_value,   # Budget ratio
            state.seller_cost / self.item_value,    # Cost ratio
        ]
        
        if state.last_buyer_offer is not None:
            features.append(state.last_buyer_offer / self.item_value)
        else:
            features.append(0.0)
            
        if state.last_seller_offer is not None:
            features.append(state.last_seller_offer / self.item_value)
        else:
            features.append(0.0)
        
        # Add agent-specific features
        if agent_type == "buyer":
            features.extend([
                state.buyer_acceptable_range[0] / self.item_value,
                state.buyer_acceptable_range[1] / self.item_value
            ])
        else:
            features.extend([
                state.seller_acceptable_range[0] / self.item_value,
                state.seller_acceptable_range[1] / self.item_value
            ])
        
        return np.array(features, dtype=np.float32)
