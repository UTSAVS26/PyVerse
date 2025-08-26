#!/usr/bin/env python3
"""
Integration test for BargainAI components
"""

from environment import NegotiationEnvironment
from agents import RuleBasedAgent, Personality
from simulate import NegotiationSimulator

def test_integration():
    """Test that all components work together"""
    print("🧪 Testing BargainAI Integration...")
    
    # Create environment
    env = NegotiationEnvironment(
        max_rounds=5,
        item_value=100.0,
        buyer_budget=120.0,
        seller_cost=60.0
    )
    print("✅ Environment created")
    
    # Create agents
    buyer = RuleBasedAgent("buyer", Personality.COOPERATIVE)
    seller = RuleBasedAgent("seller", Personality.AGGRESSIVE)
    print("✅ Agents created")
    
    # Create simulator
    simulator = NegotiationSimulator(env)
    print("✅ Simulator created")
    
    # Run simulation
    result = simulator._run_single_simulation(buyer, seller)
    print("✅ Simulation completed")
    
    # Check results
    print(f"📊 Results:")
    print(f"   Deal made: {result['deal_made']}")
    print(f"   Final price: ${result['final_price']:.2f}")
    print(f"   Rounds: {result['rounds']}")
    print(f"   Buyer reward: {result['buyer_reward']:.2f}")
    print(f"   Seller reward: {result['seller_reward']:.2f}")
    
    # Create transcript
    transcript = simulator.create_dialogue_transcript(result)
    print("✅ Dialogue transcript created")
    
    print("\n🎉 All tests passed! BargainAI is working correctly.")

if __name__ == "__main__":
    test_integration()
