import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from environment import NegotiationEnvironment
from agents import RuleBasedAgent, Personality
from simulate import NegotiationSimulator

def main():
    st.set_page_config(
        page_title="BargainAI Demo",
        page_icon="🤝",
        layout="wide"
    )
    
    st.title("🤝 BargainAI - AI Negotiation Bot Demo")
    st.markdown("Experience AI-driven negotiations in a virtual marketplace!")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Environment parameters
    st.sidebar.subheader("Environment Settings")
    item_value = st.sidebar.slider("Item Value ($)", 50, 200, 100)
    buyer_budget = st.sidebar.slider("Buyer Budget ($)", item_value, item_value + 50, item_value + 20)
    seller_cost = st.sidebar.slider("Seller Cost ($)", item_value * 0.5, item_value * 0.8, item_value * 0.6)
    max_rounds = st.sidebar.slider("Max Rounds", 5, 15, 10)
    
    # Agent personalities
    st.sidebar.subheader("Agent Personalities")
    buyer_personality = st.sidebar.selectbox(
        "Buyer Personality",
        ["cooperative", "aggressive", "deceptive", "rational"],
        index=0
    )
    
    seller_personality = st.sidebar.selectbox(
        "Seller Personality",
        ["cooperative", "aggressive", "deceptive", "rational"],
        index=1
    )
    
    # Number of simulations
    num_simulations = st.sidebar.slider("Number of Simulations", 1, 50, 10)
    
    # Create environment and agents
    env = NegotiationEnvironment(
        max_rounds=max_rounds,
        item_value=float(item_value),
        buyer_budget=float(buyer_budget),
        seller_cost=float(seller_cost)
    )
    
    buyer_agent = RuleBasedAgent("buyer", buyer_personality)
    seller_agent = RuleBasedAgent("seller", seller_personality)
    
    simulator = NegotiationSimulator(env)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 Environment Overview")
        st.metric("Item Value", f"${item_value}")
        st.metric("Buyer Budget", f"${buyer_budget}")
        st.metric("Seller Cost", f"${seller_cost}")
        st.metric("Max Rounds", max_rounds)
        
        st.subheader("👥 Agent Configuration")
        st.info(f"**Buyer**: {buyer_personality.title()}")
        st.info(f"**Seller**: {seller_personality.title()}")
    
    with col2:
        st.subheader("🎯 Negotiation Zone")
        
        # Create a simple visualization of the negotiation space
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot negotiation zones
        x = np.linspace(0, max_rounds, 100)
        
        # Buyer's acceptable range
        buyer_min = seller_cost * 0.8
        buyer_max = buyer_budget * 0.95
        ax.fill_between(x, buyer_min, buyer_max, alpha=0.3, color='blue', label='Buyer Acceptable Range')
        
        # Seller's acceptable range
        seller_min = seller_cost * 1.1
        seller_max = item_value * 0.9
        ax.fill_between(x, seller_min, seller_max, alpha=0.3, color='red', label='Seller Acceptable Range')
        
        # Reference lines
        ax.axhline(y=item_value, color='purple', linestyle='--', label='Item Value')
        ax.axhline(y=buyer_budget, color='green', linestyle='--', label='Buyer Budget')
        ax.axhline(y=seller_cost, color='orange', linestyle='--', label='Seller Cost')
        
        ax.set_xlabel('Negotiation Rounds')
        ax.set_ylabel('Price ($)')
        ax.set_title('Negotiation Space')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
    
    # Run simulations
    if st.button("🚀 Run Negotiation Simulations", type="primary"):
        with st.spinner("Running simulations..."):
            results = simulator.run_simulation(buyer_agent, seller_agent, num_simulations)
        
        st.success(f"Completed {num_simulations} simulations!")
        
        # Display results
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Success Rate", f"{results['summary']['success_rate']:.1%}")
        
        with col2:
            st.metric("Average Price", f"${results['summary']['average_price']:.2f}")
        
        with col3:
            st.metric("Average Rounds", f"{results['summary']['average_rounds']:.1f}")
        
        with col4:
            st.metric("Total Deals", results['summary']['total_deals'])
        
        # Show sample dialogue
        if results['simulations']:
            st.subheader("💬 Sample Negotiation Dialogue")
            
            # Find a successful negotiation for display
            successful_sim = None
            for sim in results['simulations']:
                if sim['deal_made']:
                    successful_sim = sim
                    break
            
            if successful_sim is None:
                successful_sim = results['simulations'][0]  # Show first one if no successful deals
            
            transcript = simulator.create_dialogue_transcript(successful_sim)
            st.text_area("Dialogue Transcript", transcript, height=400)
        
        # Create and display visualization
        st.subheader("📈 Negotiation Visualization")
        
        if results['simulations']:
            # Create visualization for the first simulation
            sample_sim = results['simulations'][0]
            
            # Create the plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Price negotiation plot
            history = sample_sim['negotiation_history']
            rounds = list(range(1, len(history) + 1))
            
            buyer_offers = []
            seller_offers = []
            
            for round_data in history:
                buyer_action = round_data['buyer_action']
                seller_action = round_data['seller_action']
                
                if buyer_action.get('action_type') == 'offer':
                    buyer_offers.append(buyer_action.get('price'))
                else:
                    buyer_offers.append(None)
                
                if seller_action.get('action_type') == 'offer':
                    seller_offers.append(seller_action.get('price'))
                else:
                    seller_offers.append(None)
            
            # Plot offers
            ax1.plot(rounds, buyer_offers, 'bo-', label='Buyer Offers', markersize=8)
            ax1.plot(rounds, seller_offers, 'ro-', label='Seller Offers', markersize=8)
            
            # Add reference lines
            ax1.axhline(y=buyer_budget, color='g', linestyle='--', alpha=0.7, label='Buyer Budget')
            ax1.axhline(y=seller_cost, color='orange', linestyle='--', alpha=0.7, label='Seller Cost')
            ax1.axhline(y=item_value, color='purple', linestyle='--', alpha=0.7, label='Item Value')
            
            if sample_sim['deal_made']:
                ax1.axhline(y=sample_sim['final_price'], color='black', linestyle='-', 
                           linewidth=2, label=f'Final Price: ${sample_sim["final_price"]:.2f}')
            
            ax1.set_xlabel('Negotiation Round')
            ax1.set_ylabel('Price ($)')
            ax1.set_title('Price Negotiation Over Time')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Action type plot
            buyer_actions = [round_data['buyer_action'].get('action_type') for round_data in history]
            seller_actions = [round_data['seller_action'].get('action_type') for round_data in history]
            
            action_types = list(set(buyer_actions + seller_actions))
            
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
            st.pyplot(fig)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    ### About BargainAI
    
    BargainAI is an experimental AI Negotiation Bot that simulates buying/selling negotiations in a virtual marketplace. 
    The bot learns strategies such as bargaining, bluffing, counter-offers, and deal acceptance through self-play simulations.
    
    **Features:**
    - 🤖 Multiple agent personalities (Cooperative, Aggressive, Deceptive, Rational)
    - 🧠 Reinforcement learning capabilities
    - 📊 Real-time negotiation visualization
    - 💬 Human-readable dialogue transcripts
    - 📈 Performance analytics and comparison
    """)

if __name__ == "__main__":
    main()
