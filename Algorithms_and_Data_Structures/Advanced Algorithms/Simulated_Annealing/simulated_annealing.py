"""
Simulated Annealing Algorithm Implementation

A probabilistic technique for approximating the global optimum of a given function.
Inspired by the annealing process in metallurgy.
"""

import random
import math
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Tuple
from dataclasses import dataclass
from copy import deepcopy


@dataclass
class OptimizationResult:
    """Result of optimization process."""
    best_solution: Any
    best_cost: float
    iterations: int
    temperature_history: List[float]
    cost_history: List[float]


class OptimizationProblem(ABC):
    """Abstract base class for optimization problems."""
    
    @abstractmethod
    def generate_initial_solution(self) -> Any:
        """Generate an initial solution."""
        pass
    
    @abstractmethod
    def generate_neighbor(self, solution: Any) -> Any:
        """Generate a neighbor solution."""
        pass
    
    @abstractmethod
    def calculate_cost(self, solution: Any) -> float:
        """Calculate the cost of a solution."""
        pass
    
    @abstractmethod
    def is_valid_solution(self, solution: Any) -> bool:
        """Check if solution is valid."""
        pass


class SimulatedAnnealing:
    """
    Simulated Annealing Algorithm Implementation
    
    Attributes:
        initial_temp (float): Initial temperature
        cooling_rate (float): Rate at which temperature decreases
        min_temp (float): Minimum temperature to stop
        max_iterations (int): Maximum number of iterations
        verbose (bool): Whether to print debug information
    """
    
    def __init__(self, initial_temp: float = 1000, cooling_rate: float = 0.95,
                 min_temp: float = 1, max_iterations: int = 10000, verbose: bool = False):
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.max_iterations = max_iterations
        self.verbose = verbose
    
    def optimize(self, problem: OptimizationProblem) -> OptimizationResult:
        """
        Perform simulated annealing optimization.
        
        Args:
            problem: Optimization problem to solve
            
        Returns:
            OptimizationResult with best solution and statistics
        """
        # Initialize
        current_solution = problem.generate_initial_solution()
        current_cost = problem.calculate_cost(current_solution)
        
        best_solution = deepcopy(current_solution)
        best_cost = current_cost
        
        temperature = self.initial_temp
        iterations = 0
        
        temperature_history = [temperature]
        cost_history = [current_cost]
        
        if self.verbose:
            print(f"Starting SA with initial cost: {current_cost}")
        
        while temperature > self.min_temp and iterations < self.max_iterations:
            # Generate neighbor
            neighbor_solution = problem.generate_neighbor(current_solution)
            neighbor_cost = problem.calculate_cost(neighbor_solution)
            
            # Calculate cost difference
            delta_cost = neighbor_cost - current_cost
            
            # Acceptance criterion
            if delta_cost < 0 or random.random() < math.exp(-delta_cost / temperature):
                current_solution = neighbor_solution
                current_cost = neighbor_cost
                
                # Update best solution if necessary
                if current_cost < best_cost:
                    best_solution = deepcopy(current_solution)
                    best_cost = current_cost
                    if self.verbose:
                        print(f"Iteration {iterations}: New best cost: {best_cost}")
            
            # Cool down
            temperature *= self.cooling_rate
            iterations += 1
            
            # Record history
            temperature_history.append(temperature)
            cost_history.append(current_cost)
            
            if self.verbose and iterations % 1000 == 0:
                print(f"Iteration {iterations}: T={temperature:.2f}, Cost={current_cost:.2f}")
        
        if self.verbose:
            print(f"Optimization complete after {iterations} iterations")
            print(f"Best cost found: {best_cost}")
        
        return OptimizationResult(
            best_solution=best_solution,
            best_cost=best_cost,
            iterations=iterations,
            temperature_history=temperature_history,
            cost_history=cost_history
        )


# Example Problem: Traveling Salesman Problem
class TSPProblem(OptimizationProblem):
    """Traveling Salesman Problem implementation."""
    
    def __init__(self, cities: List[Tuple[float, float]]):
        self.cities = cities
        self.n_cities = len(cities)
    
    def generate_initial_solution(self):
        """Generate random tour."""
        tour = list(range(self.n_cities))
        random.shuffle(tour)
        return tour
    
    def generate_neighbor(self, solution):
        """Generate neighbor using 2-opt swap."""
        neighbor = solution.copy()
        
        # Choose two random positions to swap
        i, j = random.sample(range(self.n_cities), 2)
        if i > j:
            i, j = j, i
        
        # Reverse the segment between i and j
        neighbor[i:j+1] = reversed(neighbor[i:j+1])
        
        return neighbor
    
    def calculate_cost(self, solution):
        """Calculate total tour distance."""
        total_distance = 0
        for i in range(self.n_cities):
            city1 = self.cities[solution[i]]
            city2 = self.cities[solution[(i + 1) % self.n_cities]]
            distance = math.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)
            total_distance += distance
        return total_distance
    
    def is_valid_solution(self, solution):
        """Check if tour visits all cities exactly once."""
        return len(set(solution)) == self.n_cities and len(solution) == self.n_cities


# Example Problem: Function Optimization
class FunctionOptimizationProblem(OptimizationProblem):
    """Function optimization problem."""
    
    def __init__(self, objective_function: Callable, bounds: List[Tuple[float, float]]):
        self.objective_function = objective_function
        self.bounds = bounds
        self.dimension = len(bounds)
    
    def generate_initial_solution(self):
        """Generate random point within bounds."""
        solution = []
        for lower, upper in self.bounds:
            solution.append(random.uniform(lower, upper))
        return solution
    
    def generate_neighbor(self, solution):
        """Generate neighbor by adding small random perturbation."""
        neighbor = solution.copy()
        for i in range(self.dimension):
            lower, upper = self.bounds[i]
            perturbation = random.gauss(0, (upper - lower) * 0.1)
            neighbor[i] += perturbation
            neighbor[i] = max(lower, min(upper, neighbor[i]))  # Clamp to bounds
        return neighbor
    
    def calculate_cost(self, solution):
        """Calculate function value (assuming minimization)."""
        return self.objective_function(solution)
    
    def is_valid_solution(self, solution):
        """Check if solution is within bounds."""
        for i, (lower, upper) in enumerate(self.bounds):
            if not (lower <= solution[i] <= upper):
                return False
        return True


# Example Problem: Job Scheduling
class JobSchedulingProblem(OptimizationProblem):
    """Job scheduling problem."""
    
    def __init__(self, jobs: List[Tuple[int, int]]):  # (processing_time, due_date)
        self.jobs = jobs
        self.n_jobs = len(jobs)
    
    def generate_initial_solution(self):
        """Generate random job sequence."""
        sequence = list(range(self.n_jobs))
        random.shuffle(sequence)
        return sequence
    
    def generate_neighbor(self, solution):
        """Generate neighbor by swapping two jobs."""
        neighbor = solution.copy()
        i, j = random.sample(range(self.n_jobs), 2)
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        return neighbor
    
    def calculate_cost(self, solution):
        """Calculate total tardiness."""
        total_tardiness = 0
        current_time = 0
        
        for job_idx in solution:
            processing_time, due_date = self.jobs[job_idx]
            current_time += processing_time
            tardiness = max(0, current_time - due_date)
            total_tardiness += tardiness
        
        return total_tardiness
    
    def is_valid_solution(self, solution):
        """Check if all jobs are scheduled exactly once."""
        return len(set(solution)) == self.n_jobs and len(solution) == self.n_jobs


def main():
    """Example usage of simulated annealing."""
    print("=== Simulated Annealing Algorithm Demo ===\n")
    
    # Example 1: Traveling Salesman Problem
    print("1. Traveling Salesman Problem:")
    cities = [
        (0, 0), (1, 2), (3, 1), (4, 3), (5, 0),
        (6, 2), (7, 4), (8, 1), (9, 3), (10, 0)
    ]
    
    tsp_problem = TSPProblem(cities)
    sa = SimulatedAnnealing(initial_temp=100, cooling_rate=0.99, 
                           min_temp=0.1, max_iterations=5000, verbose=True)
    
    result = sa.optimize(tsp_problem)
    print(f"Best tour cost: {result.best_cost:.2f}")
    print(f"Best tour: {result.best_solution}")
    
    print("\n" + "="*50 + "\n")
    
    # Example 2: Function Optimization
    print("2. Function Optimization (Rastrigin Function):")
    
    def rastrigin_function(x):
        """Rastrigin function for testing optimization."""
        n = len(x)
        A = 10
        return A * n + sum(xi**2 - A * np.cos(2 * np.pi * xi) for xi in x)
    
    bounds = [(-5.12, 5.12), (-5.12, 5.12)]  # 2D optimization
    func_problem = FunctionOptimizationProblem(rastrigin_function, bounds)
    
    sa = SimulatedAnnealing(initial_temp=100, cooling_rate=0.95,
                           min_temp=0.01, max_iterations=3000, verbose=True)
    
    result = sa.optimize(func_problem)
    print(f"Best function value: {result.best_cost:.6f}")
    print(f"Best solution: {result.best_solution}")
    
    print("\n" + "="*50 + "\n")
    
    # Example 3: Job Scheduling
    print("3. Job Scheduling Problem:")
    jobs = [
        (3, 5),   # (processing_time, due_date)
        (2, 3),
        (4, 7),
        (1, 2),
        (5, 10)
    ]
    
    job_problem = JobSchedulingProblem(jobs)
    sa = SimulatedAnnealing(initial_temp=50, cooling_rate=0.98,
                           min_temp=0.1, max_iterations=2000, verbose=True)
    
    result = sa.optimize(job_problem)
    print(f"Minimum total tardiness: {result.best_cost}")
    print(f"Optimal job sequence: {result.best_solution}")
    
    # Calculate schedule details
    current_time = 0
    print("\nOptimal Schedule:")
    for job_idx in result.best_solution:
        processing_time, due_date = jobs[job_idx]
        start_time = current_time
        finish_time = current_time + processing_time
        tardiness = max(0, finish_time - due_date)
        print(f"Job {job_idx}: Start={start_time}, Finish={finish_time}, "
              f"Due={due_date}, Tardiness={tardiness}")
        current_time = finish_time


if __name__ == "__main__":
    main() 