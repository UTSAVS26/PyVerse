# Monte Carlo Tree Search (MCTS)

## Overview

Monte Carlo Tree Search (MCTS) is a heuristic search algorithm for some kinds of decision processes, particularly those employed in game playing. It combines the generality of random sampling with the precision of tree search.

## How it Works

MCTS consists of four main steps that are repeated until a computational budget is reached:

1. **Selection**: Start from root and select child nodes until a leaf node is reached
2. **Expansion**: Create one or more child nodes from the leaf node
3. **Simulation**: Perform a random playout from the new node to a terminal state
4. **Backpropagation**: Update the statistics of all nodes along the path

## Key Components

### Selection Strategy
- **UCB1 (Upper Confidence Bound)**: Balances exploration and exploitation
- **UCB1-Tuned**: Improved version of UCB1
- **Progressive Bias**: Incorporates domain knowledge

### Expansion Policy
- **Single Child**: Expand one child at a time
- **All Children**: Expand all possible children
- **Progressive Widening**: Expand based on visit count

### Simulation Policy
- **Random**: Completely random playout
- **Heuristic**: Use domain knowledge for better playouts
- **Light Playout**: Fast approximation of game state

### Backpropagation
- **Win/Loss**: Binary outcome
- **Score**: Continuous outcome
- **Draw**: Special handling for ties

## Applications

- Game playing (Go, Chess, Checkers)
- Planning and decision making
- Robotics path planning
- Resource allocation
- Anomaly detection

## Algorithm Steps

```
1. Initialize: Create root node
2. While computational budget not exhausted:
   a. Selection: Choose path from root to leaf using UCB1
   b. Expansion: Add child node to leaf
   c. Simulation: Random playout from new node
   d. Backpropagation: Update statistics along path
3. Return best child of root
```

## Time Complexity

- **Per Iteration**: O(depth * simulation_cost)
- **Total**: O(iterations * depth * simulation_cost)

## Space Complexity

- O(nodes_in_tree) where nodes grow with iterations

## Usage

```python
from mcts import MCTS

# Define your game/problem
game = YourGame()

# Create MCTS instance
mcts = MCTS(
    exploration_constant=1.414,
    simulation_count=1000,
    time_limit=1.0
)

# Find best action
best_action = mcts.search(game)
```

## Parameters

- **Exploration Constant**: Controls exploration vs exploitation
- **Simulation Count**: Number of playouts per iteration
- **Time Limit**: Maximum time to spend on search
- **Iteration Limit**: Maximum number of iterations
- **UCB1 Formula**: sqrt(ln(parent_visits) / child_visits)

## Example Problems

1. **Tic-Tac-Toe**: Simple game with clear win/loss conditions
2. **Connect Four**: Strategic game with multiple winning patterns
3. **2048**: Single-player puzzle game
4. **Path Planning**: Find optimal path in grid world
5. **Resource Allocation**: Optimize resource distribution

## Advantages

- No domain knowledge required
- Anytime algorithm (can be stopped early)
- Parallelizable
- Handles large state spaces
- Robust and simple to implement

## Disadvantages

- Requires many simulations
- May not find optimal solution
- Memory usage grows with tree size
- Sensitive to simulation quality 