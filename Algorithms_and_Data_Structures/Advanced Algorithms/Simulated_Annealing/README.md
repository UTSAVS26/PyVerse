# Simulated Annealing Algorithm

## Overview

Simulated Annealing is a probabilistic technique for approximating the global optimum of a given function. It's inspired by the annealing process in metallurgy, where a material is heated and then slowly cooled to reduce defects and achieve a more stable crystalline structure.

## How it Works

1. **Initialization**: Start with an initial solution and high temperature
2. **Neighbor Generation**: Generate a neighbor solution by making a small change
3. **Acceptance Criterion**: Accept the neighbor if it's better, or with probability based on temperature and cost difference
4. **Temperature Reduction**: Gradually reduce temperature according to cooling schedule
5. **Termination**: Stop when temperature is low enough or maximum iterations reached

## Key Parameters

- **Initial Temperature**: Starting temperature (high)
- **Cooling Rate**: How fast temperature decreases
- **Neighbor Function**: How to generate neighboring solutions
- **Cost Function**: How to evaluate solution quality
- **Acceptance Probability**: P(accept) = exp(-ΔE/T)

## Applications

- Traveling Salesman Problem (TSP)
- Job scheduling
- Circuit design
- Network design
- Function optimization
- Machine learning hyperparameter tuning

## Algorithm Steps

```
1. Initialize: solution = random_solution(), T = T_initial
2. While T > T_min and iterations < max_iterations:
   a. Generate neighbor = neighbor_function(solution)
   b. ΔE = cost(neighbor) - cost(solution)
   c. If ΔE < 0 or random() < exp(-ΔE/T):
      solution = neighbor
   d. T = cooling_schedule(T)
3. Return best_solution_found
```

## Time Complexity

- **Best Case**: O(n) where n is number of iterations
- **Worst Case**: O(n * neighbor_generation_cost)

## Space Complexity

- O(1) for basic implementation
- O(n) if storing all visited solutions

## Usage

```python
from simulated_annealing import SimulatedAnnealing

# Define your problem
problem = YourOptimizationProblem()

# Create SA instance
sa = SimulatedAnnealing(
    initial_temp=1000,
    cooling_rate=0.95,
    min_temp=1,
    max_iterations=10000
)

# Find optimal solution
best_solution = sa.optimize(problem)
```

## Cooling Schedules

1. **Linear**: T = T_initial - α * iteration
2. **Exponential**: T = T_initial * α^iteration
3. **Logarithmic**: T = T_initial / log(1 + iteration)
4. **Geometric**: T = T_initial * α^iteration

## Example Problems

1. **Traveling Salesman Problem**: Find shortest tour visiting all cities
2. **Function Optimization**: Find global minimum of complex functions
3. **Job Scheduling**: Optimize task assignment to minimize completion time
4. **Graph Coloring**: Find valid coloring with minimum colors 