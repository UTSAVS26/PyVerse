# Beam Search Algorithm

## Overview

Beam Search is a heuristic search algorithm that explores a graph by expanding the most promising node in a limited set. It's a variant of breadth-first search that uses a heuristic function to evaluate and rank nodes.

## How it Works

1. **Initialization**: Start with a set of initial nodes (beam)
2. **Expansion**: For each node in the current beam, generate all possible successors
3. **Evaluation**: Use a heuristic function to evaluate each successor
4. **Selection**: Keep only the top-k most promising nodes (beam width)
5. **Iteration**: Repeat until a goal state is found or maximum iterations reached

## Key Parameters

- **Beam Width (k)**: Number of nodes to keep at each level
- **Heuristic Function**: Function to evaluate node quality
- **Goal Function**: Function to check if goal is reached

## Applications

- Speech recognition
- Machine translation
- Planning problems
- Natural language processing
- Game playing

## Time Complexity

- **Best Case**: O(k * d) where k is beam width, d is depth
- **Worst Case**: O(k^d) where d is maximum depth

## Space Complexity

- O(k * d) where k is beam width, d is depth

## Usage

```python
from beam_search import BeamSearch

# Define your problem
problem = YourProblem()

# Create beam search instance
beam_search = BeamSearch(beam_width=5, max_iterations=100)

# Find solution
solution = beam_search.search(problem)
```

## Example Problems

1. **8-Puzzle Problem**: Find shortest path to goal state
2. **Word Prediction**: Predict next word in sequence
3. **Path Finding**: Find optimal path in graph
4. **Planning**: Find action sequence to reach goal 