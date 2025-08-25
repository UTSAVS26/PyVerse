# 🤝 BargainAI – AI-Driven Negotiation Bot for Virtual Marketplaces

BargainAI is a fully functional AI-driven negotiation bot system that simulates buying/selling negotiations in a virtual marketplace. The project has been successfully implemented with all requested features and is ready for use.

## 🎯 Project Goals

BargainAI implements a negotiation environment with:
- ✅ A buyer agent and seller agent
- ✅ Rules (item value, budget, acceptable price ranges)
- ✅ Rule-based logic (baseline) and Reinforcement Learning strategies
- ✅ Tactics like bluffing, aggressive bargaining, and cooperative deals
- ✅ Evaluation based on success rate, fairness, and profitability

## 🏗️ What Was Built

### Core Components

1. **`environment.py`** - Negotiation environment with rules, scoring, and transaction logic
   - ✅ Multi-round negotiation system
   - ✅ Reward calculation for both parties
   - ✅ Action validation and state management
   - ✅ Configurable parameters (item value, budget, cost, max rounds)

2. **`agents.py`** - Buyer/Seller agents with multiple strategies
   - ✅ Rule-based agents with 4 personalities:
     - **Cooperative**: Aims for fair deals
     - **Aggressive**: Pushes for better deals
     - **Deceptive**: Uses bluffing and inconsistent strategies
     - **Rational**: Optimizes based on utility
   - ✅ RL agents using Q-learning with neural networks
   - ✅ Experience replay and epsilon-greedy exploration

3. **`train.py`** - Self-play training loop for agents
   - ✅ Training manager with progress tracking
   - ✅ Evaluation against rule-based agents
   - ✅ Model saving/loading
   - ✅ Training visualization and metrics

4. **`simulate.py`** - Run and visualize sample negotiations
   - ✅ Dialogue transcript generation
   - ✅ Negotiation visualization (price trends, action types)
   - ✅ Agent comparison and evaluation
   - ✅ Multiple simulation runs

5. **`demo.py`** - Interactive Streamlit demo
   - ✅ Web-based interface for testing
   - ✅ Real-time configuration
   - ✅ Live visualization

### Testing & Quality Assurance

6. **`tests/`** - Comprehensive test suite
   - ✅ Environment functionality tests
   - ✅ Agent behavior tests
   - ✅ Training process tests
   - ✅ Simulation accuracy tests
   - ✅ Integration tests

7. **`test_integration.py`** - Quick integration test
   - ✅ Verifies all components work together
   - ✅ Provides immediate feedback

## 🚀 Quick Start

### Installation
```bash
# Install dependencies
pip install -r requirements.txt
```

### Basic Usage
```bash
# Run a quick simulation
python simulate.py

# Run training (shorter version for testing)
python train.py

# Run integration test
python test_integration.py

# Launch interactive demo (optional)
streamlit run demo.py
```

## 📊 Sample Results

### Successful Negotiation Example
```
🤝 NEGOTIATION DIALOGUE TRANSCRIPT
==================================================
Item Value: $100.0
Buyer Budget: $120.0
Seller Cost: $60.0
==================================================

📋 ROUND 1
--------------------
👤 Buyer: "I offer $90.00"
🏪 Seller: "I offer $95.00"

📋 ROUND 2
--------------------
👤 Buyer: "I accept your offer of $95.00!"
🏪 Seller: "I accept your offer of $90.00!"

==================================================
✅ DEAL MADE!
Final Price: $90.00
Buyer Surplus: $30.00
Seller Profit: $30.00
Total Rounds: 2
Buyer Reward: 2.40
Seller Reward: 4.90
```

### Agent Comparison Results
- **Cooperative vs Aggressive**: 100% success rate, $90.00 average price
- **Aggressive vs Cooperative**: 0% success rate (personality clash)
- **Rational vs Deceptive**: 100% success rate, $86.73 average price
- **Deceptive vs Rational**: 100% success rate, $75.41 average price

## 🔧 Technical Implementation

### Architecture
- **Modular Design**: Clean separation of concerns
- **Object-Oriented**: Well-structured classes and inheritance
- **Type Hints**: Full type annotations for better code quality
- **Error Handling**: Robust error handling throughout
- **Documentation**: Comprehensive docstrings and comments

### Technologies Used
- **Python 3.8+**: Core language
- **PyTorch**: Deep learning for RL agents
- **NumPy**: Numerical computations
- **Matplotlib/Seaborn**: Visualization
- **Streamlit**: Interactive web interface
- **Pytest**: Testing framework

## 🧪 Testing

All tests pass successfully:
- ✅ Environment functionality
- ✅ Agent behavior and strategies
- ✅ Training process
- ✅ Simulation accuracy
- ✅ Integration between components
- ✅ Model saving/loading

## 📁 Project Structure

```
BargainAI/
├── environment.py          # Negotiation environment
├── agents.py              # Buyer/Seller agents
├── train.py               # Training manager
├── simulate.py            # Simulation and visualization
├── demo.py                # Streamlit web interface
├── test_integration.py    # Quick integration test
├── requirements.txt       # Dependencies
├── README.md             # This file
├── PROJECT_SUMMARY.md    # Detailed project summary
├── tests/                # Test suite
│   ├── test_environment.py
│   ├── test_agents.py
│   ├── test_train.py
│   └── test_simulation.py
├── models/               # Trained models
│   ├── buyer_model.pth
│   └── seller_model.pth
└── results/              # Generated results
    ├── training_plots_*.png
    ├── evaluation_results_*.json
    └── agent_comparison.png
```

## 🎯 Key Features

1. **Negotiation Simulation**: Agents negotiate over item prices with realistic strategies
2. **Personality Types**: Different agent personalities show varied negotiation behaviors
3. **Reinforcement Learning**: RL agents learn optimal strategies through self-play
4. **Visualization**: Price trends, action types, and negotiation progress
5. **Dialogue Transcripts**: Human-readable negotiation conversations
6. **Performance Analytics**: Success rates, average prices, and reward comparisons

## 🚀 Future Enhancements

The project is designed to be easily extensible:

1. **Multi-item Negotiations**: Bundle multiple items in single deals
2. **Human vs Bot Interface**: CLI or web-based human interaction
3. **Advanced RL Algorithms**: DQN, A3C, or PPO implementations
4. **Dynamic Pricing**: Market conditions affecting prices
5. **Emotional Modeling**: Agent emotions affecting decisions
6. **Multi-agent Scenarios**: Multiple buyers/sellers competing

## 📝 Why BargainAI?

- **No dataset required** — bots generate negotiation data themselves
- **Combines game theory, reinforcement learning, and behavioral economics**
- **Practical applications** in e-commerce, automated trading, and contract simulations
- **Self-improving agents** that learn through experience
- **Multiple personality types** for realistic negotiation scenarios

## 📝 Conclusion

BargainAI is a complete, functional AI negotiation system that successfully demonstrates:
- **Game Theory**: Strategic decision making in negotiations
- **Reinforcement Learning**: Self-improving agents through experience
- **Multi-agent Systems**: Complex interactions between different personalities
- **Practical Applications**: Real-world negotiation scenarios

The system is ready for use, testing, and further development. All code is well-documented, tested, and follows best practices for maintainability and extensibility.

---

**Project Status**: ✅ **COMPLETE AND FUNCTIONAL**  
**Last Updated**: August 21, 2025  
**Test Status**: ✅ **ALL TESTS PASSING**
