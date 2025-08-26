# 🧠 Autonomous Snake Navigation System - Advanced Reinforcement Learning Framework

A Python implementation of the classic Snake game with reinforcement learning agents (Q-Learning and Deep Q-Network) to train AI to play optimally.

## 🎯 Project Overview

This project implements:
- **Snake Game Environment**: A Pygame-based game environment with configurable grid sizes
- **Q-Learning Agent**: Tabular Q-learning for smaller state spaces
- **Deep Q-Network (DQN)**: Neural network-based agent for larger state spaces
- **Training Pipeline**: Complete training loop with metrics and visualization
- **Testing Suite**: Comprehensive tests for all components

## 🚀 Features

- **Flexible Environment**: Configurable grid sizes, reward functions, and game parameters
- **Multiple RL Agents**: Both Q-Learning and DQN implementations
- **Real-time Visualization**: Watch the agent learn and play
- **Comprehensive Logging**: Track training progress and performance metrics
- **Extensive Testing**: Unit tests for all components

## 📋 Requirements

- Python 3.8+
- Pygame
- PyTorch
- NumPy
- Matplotlib
- Seaborn

## 🛠️ Installation

1. **Clone the repository**:

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## 🎮 Usage

### Quick Start - Watch AI Play

```bash
python main.py --mode train --agent dqn --episodes 1000
```

### Train Q-Learning Agent

```bash
python main.py --mode train --agent qlearning --episodes 500 --grid-size 10
```

### Train DQN Agent

```bash
python main.py --mode train --agent dqn --episodes 2000 --grid-size 15
```

### Watch Trained Agent

```bash
python main.py --mode play --agent dqn --model-path models/dqn_model.pth
```

### Run Tests

```bash
python -m pytest tests/ -v
```

## 📊 Training Results

The agents typically achieve:
- **Q-Learning**: 15-25 average score on 10x10 grid
- **DQN**: 30-50 average score on 15x15 grid

Training progress is logged and visualized with:
- Average score per episode
- Reward progression
- Learning curves

## 🏗️ Project Structure

```
SnakeRL/
├── src/
│   ├── environment/
│   │   ├── __init__.py
│   │   ├── snake_env.py          # Snake game environment
│   │   └── state_representation.py # State encoding utilities
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base_agent.py         # Abstract base agent
│   │   ├── qlearning_agent.py    # Q-Learning implementation
│   │   └── dqn_agent.py          # Deep Q-Network implementation
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── visualization.py      # Plotting and visualization
│   │   └── logger.py             # Logging utilities
│   └── models/
│       ├── __init__.py
│       └── dqn_model.py          # Neural network architecture
├── tests/
│   ├── __init__.py
│   ├── test_environment.py       # Environment tests
│   ├── test_agents.py            # Agent tests
│   └── test_utils.py             # Utility tests
├── main.py                       # Main training script
├── requirements.txt              # Dependencies
├── README.md                     # This file
└── .gitignore                    # Git ignore file
```

## 🧪 Testing

The project includes comprehensive tests:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_environment.py -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

## 📈 Performance Metrics

- **Average Score**: Mean score over last 100 episodes
- **Best Score**: Highest score achieved during training
- **Success Rate**: Percentage of episodes with score > 0
- **Training Time**: Time taken to complete training

## 🔧 Configuration

Key parameters can be adjusted in the training script:
- Grid size (default: 10x10)
- Number of episodes (default: 1000)
- Learning rate (default: 0.1 for Q-Learning, 0.001 for DQN)
- Epsilon decay (default: 0.995)
- Reward values (food: +10, collision: -10, step: -0.1)