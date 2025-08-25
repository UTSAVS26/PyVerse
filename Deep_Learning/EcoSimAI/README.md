# 🌱 EcoSimAI – Virtual Ecosystem Simulation with Evolving RL Agents

EcoSimAI is a sophisticated virtual ecosystem simulation where intelligent agents (predators, prey, and plants) interact in a shared 2D environment. Each agent has survival goals: predators hunt prey, prey search for food and avoid predators, and plants grow and spread. Using Reinforcement Learning (RL), agents evolve strategies to maximize survival and reproduction in a procedurally generated, ever-changing environment.

## 🎯 Project Goals

- **Build a 2D grid-based ecosystem** with plants, prey, and predators
- **Implement intelligent agent behavior** using both rule-based and RL approaches
- **Track ecosystem dynamics** including population balance and extinction events
- **Visualize the simulation** in real-time with moving agents
- **Demonstrate emergent behavior** through AI adaptation in dynamic environments

## 🏗️ Architecture

### Core Components

- **`environment.py`** - Ecosystem simulation engine managing the 2D grid and agent interactions
- **`agents.py`** - Agent classes (Predator, Prey, Plant) with behavior logic
- **`rl_agent.py`** - Reinforcement Learning implementation with Q-learning and DQN
- **`simulate.py`** - Main simulation runner with visualization
- **`test_*.py`** - Comprehensive test suite for all components

### Agent Types

1. **Plants** 🌿
   - Grow and spread to nearby empty cells
   - Provide food for prey
   - Regenerate over time

2. **Prey** 🦌
   - Consume plants for energy
   - Avoid predators using vision and movement
   - Reproduce when energy is high
   - Can use RL for adaptive behavior

3. **Predators** 🐺
   - Hunt prey for energy
   - Use vision to locate targets
   - Reproduce when energy is high
   - Can use RL for strategic hunting

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd EcoSimAI
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run a basic simulation:**
   ```bash
   python simulate.py --steps 1000 --speed 0.1
   ```

### Basic Usage

```python
from environment import Environment
from agents import Prey, Predator
from rl_agent import RLPrey, RLPredator

# Create environment
env = Environment(width=50, height=50)

# Run simulation
for step in range(1000):
    env.step()
    stats = env.get_statistics()
    print(f"Step {step}: Plants={stats['plants']}, Prey={stats['prey']}, Predators={stats['predators']}")
```

## 🎮 Simulation Controls

### Command Line Options

```bash
# Basic simulation
python simulate.py --steps 1000

# Custom environment size
python simulate.py --width 100 --height 100

# Adjust simulation speed
python simulate.py --speed 0.05  # Faster
python simulate.py --speed 0.5   # Slower

# Use RL agents
python simulate.py --rl-prey 0.5 --rl-predators 0.3

# Headless mode (no visualization)
python simulate.py --headless

# Use matplotlib instead of pygame
python simulate.py --matplotlib

# Save statistics to CSV
python simulate.py --save-stats
```

### Visualization Options

- **Pygame** (default): Real-time grid visualization with statistics
- **Matplotlib**: Grid view with population trend graphs
- **Headless**: No visualization, fastest execution

## 🧠 Reinforcement Learning Features

### RL Agent Capabilities

- **State Representation**: 13-dimensional vector including:
  - Agent energy and age
  - Position information
  - Nearby entity counts
  - Direction to nearest entities

- **Action Space**: 9 possible actions (8 directions + stay)

- **Learning Algorithms**:
  - **Q-Learning**: Table-based approach for simple scenarios
  - **Deep Q-Network (DQN)**: Neural network-based learning

### Reward Systems

**Prey Rewards:**
- ✅ +10 for eating plants
- ✅ +1 for staying alive
- ✅ +2 for high energy
- ❌ -5 for being near predators
- ❌ -5 for low energy

**Predator Rewards:**
- ✅ +3 for being near prey
- ✅ +1 for staying alive
- ✅ +2 for high energy
- ❌ -5 for low energy

## 📊 Statistics and Analysis

### Tracked Metrics

- **Population counts** for each agent type
- **Energy levels** and survival rates
- **Reproduction events** and offspring counts
- **Extinction events** and ecosystem stability

### Data Export

```python
# Save simulation statistics
from simulate import save_simulation_stats

stats_list = [...]  # List of statistics dictionaries
save_simulation_stats(stats_list, "ecosystem_data.csv")
```

## 🧪 Testing

### Run All Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test files
pytest test_environment.py
pytest test_agents.py
pytest test_rl_agent.py
pytest test_simulate.py
```

### Test Categories

- **Unit Tests**: Individual component functionality
- **Integration Tests**: Component interactions
- **Performance Tests**: Simulation speed and memory usage
- **Visualization Tests**: Display and rendering functionality

## 🔧 Configuration

### Environment Parameters

```python
env = Environment(
    width=50,                    # Grid width
    height=50,                   # Grid height
    plant_growth_rate=0.1,       # Plant growth probability
    initial_plants=100,          # Starting plant count
    initial_prey=50,             # Starting prey count
    initial_predators=20         # Starting predator count
)
```

### Agent Parameters

```python
# Prey configuration
prey = Prey(
    agent_id=0,
    position=Position(5, 5),
    energy=50,                   # Starting energy
    # Additional parameters inherited from Agent class
)

# RL Agent configuration
rl_prey = RLPrey(
    agent_id=0,
    position=Position(5, 5),
    energy=50,
    learning_rate=0.001,         # Neural network learning rate
    epsilon=0.1,                 # Exploration rate
    gamma=0.95                   # Discount factor
)
```

## 🎯 Advanced Features

### Nice-to-Have Implementations

- ✅ **Multiple species** with different abilities
- ✅ **Evolutionary mechanics** (agents inherit traits)
- ✅ **Visualization dashboard** with population graphs
- ✅ **Pause/Resume** functionality
- ✅ **Adjustable simulation speed**

### Future Enhancements

- 🌱 **Seasonal changes** (resource scarcity, harsh winters)
- 🧬 **Genetic algorithms** for trait evolution
- 🌍 **Multiple environments** with different conditions
- 📈 **Advanced analytics** and prediction models
- 🎨 **Enhanced visualization** with agent trails and heatmaps

## 📈 Example Results

### Typical Simulation Outcomes

1. **Balanced Ecosystem**: Stable populations with periodic fluctuations
2. **Predator Dominance**: Predators overhunt, leading to prey extinction
3. **Prey Dominance**: Prey overgraze plants, leading to ecosystem collapse
4. **RL Adaptation**: Agents develop sophisticated survival strategies

### Performance Metrics

- **Simulation Speed**: ~1000 steps/second (headless mode)
- **Memory Usage**: ~50MB for 50x50 grid with 100 agents
- **Learning Convergence**: RL agents typically show improvement within 100-500 steps

## 🤝 Contributing

### Development Setup

1. **Fork the repository**
2. **Create a feature branch**
3. **Write tests** for new functionality
4. **Ensure all tests pass**
5. **Submit a pull request**

### Code Style

- Follow PEP 8 guidelines
- Add type hints for function parameters
- Include docstrings for all classes and methods
- Write comprehensive tests for new features

## 📚 Educational Value

EcoSimAI demonstrates key concepts in:

- **Reinforcement Learning**: Q-learning, DQN, experience replay
- **Game Theory**: Competitive and cooperative behaviors
- **Complexity Science**: Emergent behavior from simple rules
- **Ecology**: Population dynamics and ecosystem stability
- **Artificial Intelligence**: Adaptive behavior in dynamic environments

## 🐛 Troubleshooting

### Common Issues

1. **Pygame not available**: Use `--matplotlib` flag or install pygame
2. **Slow performance**: Use `--headless` mode for faster execution
3. **Memory issues**: Reduce grid size or agent count
4. **RL not learning**: Check reward functions and hyperparameters

### Performance Tips

- Use headless mode for long simulations
- Reduce visualization frequency for better performance
- Adjust batch sizes for RL agents based on available memory
- Use smaller grid sizes for testing

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Inspired by classic ecosystem simulations like Conway's Game of Life
- Built on modern RL frameworks (PyTorch)
- Visualization powered by Pygame and Matplotlib
- Testing framework provided by pytest

---

**Happy simulating! 🌱🦌🐺**
