"""
Genetic Algorithm Framework Implementation

A metaheuristic inspired by the process of natural selection that uses
techniques such as inheritance, mutation, selection, and crossover.
"""

import random
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Any, Callable, Tuple, Optional
from dataclasses import dataclass
from copy import deepcopy
from enum import Enum


class SelectionMethod(Enum):
    """Selection methods for genetic algorithm."""
    ROULETTE_WHEEL = "roulette_wheel"
    TOURNAMENT = "tournament"
    RANK_BASED = "rank_based"


class CrossoverMethod(Enum):
    """Crossover methods for genetic algorithm."""
    SINGLE_POINT = "single_point"
    TWO_POINT = "two_point"
    UNIFORM = "uniform"
    ORDER_CROSSOVER = "order_crossover"
    PMX = "pmx"


class MutationMethod(Enum):
    """Mutation methods for genetic algorithm."""
    BIT_FLIP = "bit_flip"
    SWAP = "swap"
    INVERSION = "inversion"
    GAUSSIAN = "gaussian"


@dataclass
class Individual:
    """Represents an individual in the population."""
    genotype: Any
    fitness: float = 0.0
    
    def __lt__(self, other):
        return self.fitness < other.fitness


@dataclass
class EvolutionResult:
    """Result of genetic algorithm evolution."""
    best_individual: Individual
    best_fitness: float
    generation: int
    population: List[Individual]
    fitness_history: List[float]


class GeneticProblem(ABC):
    """Abstract base class for genetic algorithm problems."""
    
    @abstractmethod
    def generate_individual(self) -> Any:
        """Generate a random individual."""
        pass
    
    @abstractmethod
    def calculate_fitness(self, individual: Any) -> float:
        """Calculate fitness of an individual."""
        pass
    
    @abstractmethod
    def crossover(self, parent1: Any, parent2: Any) -> Tuple[Any, Any]:
        """Perform crossover between two parents."""
        pass
    
    @abstractmethod
    def mutate(self, individual: Any) -> Any:
        """Perform mutation on an individual."""
        pass
    
    @abstractmethod
    def is_valid(self, individual: Any) -> bool:
        """Check if individual is valid."""
        pass


class GeneticAlgorithm:
    """
    Genetic Algorithm Implementation
    
    Attributes:
        population_size (int): Size of population
        generations (int): Number of generations
        mutation_rate (float): Probability of mutation
        crossover_rate (float): Probability of crossover
        elitism_rate (float): Fraction of best individuals to preserve
        selection_method (SelectionMethod): Method for parent selection
        verbose (bool): Whether to print debug information
    """
    
    def __init__(self, population_size: int = 100, generations: int = 1000,
                 mutation_rate: float = 0.01, crossover_rate: float = 0.8,
                 elitism_rate: float = 0.1, selection_method: SelectionMethod = SelectionMethod.TOURNAMENT,
                 verbose: bool = False):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_rate = elitism_rate
        self.selection_method = selection_method
        self.verbose = verbose
    
    def evolve(self, problem: GeneticProblem) -> EvolutionResult:
        """
        Evolve population to find optimal solution.
        
        Args:
            problem: Genetic problem to solve
            
        Returns:
            EvolutionResult with best individual and statistics
        """
        # Initialize population
        population = self._initialize_population(problem)
        self._evaluate_population(population, problem)
        
        best_fitness_history = []
        best_individual = max(population, key=lambda x: x.fitness)
        best_fitness_history.append(best_individual.fitness)
        
        if self.verbose:
            print(f"Initial best fitness: {best_individual.fitness}")
        
        # Evolution loop
        for generation in range(self.generations):
            # Selection
            parents = self._select_parents(population)
            
            # Crossover
            offspring = self._crossover(parents, problem)
            
            # Mutation
            self._mutate(offspring, problem)
            
            # Evaluation
            self._evaluate_population(offspring, problem)
            
            # Replacement with elitism
            population = self._replace_population(population, offspring)
            
            # Update best individual
            current_best = max(population, key=lambda x: x.fitness)
            if current_best.fitness > best_individual.fitness:
                best_individual = deepcopy(current_best)
            
            best_fitness_history.append(best_individual.fitness)
            
            if self.verbose and generation % 100 == 0:
                print(f"Generation {generation}: Best fitness = {best_individual.fitness}")
        
        if self.verbose:
            print(f"Evolution complete. Best fitness: {best_individual.fitness}")
        
        return EvolutionResult(
            best_individual=best_individual,
            best_fitness=best_individual.fitness,
            generation=self.generations,
            population=population,
            fitness_history=best_fitness_history
        )
    
    def _initialize_population(self, problem: GeneticProblem) -> List[Individual]:
        """Initialize random population."""
        population = []
        for _ in range(self.population_size):
            genotype = problem.generate_individual()
            individual = Individual(genotype=genotype)
            population.append(individual)
        return population
    
    def _evaluate_population(self, population: List[Individual], problem: GeneticProblem):
        """Evaluate fitness of all individuals in population."""
        for individual in population:
            individual.fitness = problem.calculate_fitness(individual.genotype)
    
    def _select_parents(self, population: List[Individual]) -> List[Individual]:
        """Select parents for reproduction."""
        parents = []
        for _ in range(self.population_size):
            if self.selection_method == SelectionMethod.ROULETTE_WHEEL:
                parent = self._roulette_wheel_selection(population)
            elif self.selection_method == SelectionMethod.TOURNAMENT:
                parent = self._tournament_selection(population)
            elif self.selection_method == SelectionMethod.RANK_BASED:
                parent = self._rank_based_selection(population)
            else:
                parent = self._tournament_selection(population)
            parents.append(parent)
        return parents
    
    def _roulette_wheel_selection(self, population: List[Individual]) -> Individual:
        """Roulette wheel selection."""
        total_fitness = sum(ind.fitness for ind in population)
        if total_fitness == 0:
            return random.choice(population)
        
        r = random.uniform(0, total_fitness)
        cumulative_fitness = 0
        for individual in population:
            cumulative_fitness += individual.fitness
            if cumulative_fitness >= r:
                return individual
        return population[-1]
    
    def _tournament_selection(self, population: List[Individual], tournament_size: int = 3) -> Individual:
        """Tournament selection."""
        tournament = random.sample(population, tournament_size)
        return max(tournament, key=lambda x: x.fitness)
    
    def _rank_based_selection(self, population: List[Individual]) -> Individual:
        """Rank-based selection."""
        sorted_population = sorted(population, key=lambda x: x.fitness)
        n = len(sorted_population)
        rank_weights = [i + 1 for i in range(n)]
        total_weight = sum(rank_weights)
        
        r = random.uniform(0, total_weight)
        cumulative_weight = 0
        for i, weight in enumerate(rank_weights):
            cumulative_weight += weight
            if cumulative_weight >= r:
                return sorted_population[i]
        return sorted_population[-1]
    
    def _crossover(self, parents: List[Individual], problem: GeneticProblem) -> List[Individual]:
        """Perform crossover to create offspring."""
        offspring = []
        
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                parent1, parent2 = parents[i], parents[i + 1]
                
                if random.random() < self.crossover_rate:
                    child1_genotype, child2_genotype = problem.crossover(
                        parent1.genotype, parent2.genotype
                    )
                    offspring.extend([
                        Individual(genotype=child1_genotype),
                        Individual(genotype=child2_genotype)
                    ])
                else:
                    offspring.extend([deepcopy(parent1), deepcopy(parent2)])
            else:
                offspring.append(deepcopy(parents[i]))
        
        return offspring[:self.population_size]
    
    def _mutate(self, offspring: List[Individual], problem: GeneticProblem):
        """Apply mutation to offspring."""
        for individual in offspring:
            if random.random() < self.mutation_rate:
                individual.genotype = problem.mutate(individual.genotype)
    
    def _replace_population(self, old_population: List[Individual], 
                          new_population: List[Individual]) -> List[Individual]:
        """Replace population with elitism."""
        # Sort by fitness
        old_population.sort(key=lambda x: x.fitness, reverse=True)
        new_population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Calculate number of elite individuals
        elite_count = int(self.population_size * self.elitism_rate)
        
        # Keep best individuals from old population
        elite = old_population[:elite_count]
        
        # Take remaining from new population
        remaining = new_population[:(self.population_size - elite_count)]
        
        return elite + remaining


# Example Problem: Binary String Optimization
class BinaryStringProblem(GeneticProblem):
    """Binary string optimization problem."""
    
    def __init__(self, string_length: int = 20, target_ones: int = 10):
        self.string_length = string_length
        self.target_ones = target_ones
    
    def generate_individual(self):
        """Generate random binary string."""
        return [random.randint(0, 1) for _ in range(self.string_length)]
    
    def calculate_fitness(self, individual):
        """Calculate fitness based on number of ones."""
        ones_count = sum(individual)
        return -abs(ones_count - self.target_ones)  # Negative for minimization
    
    def crossover(self, parent1, parent2):
        """Single point crossover."""
        point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    
    def mutate(self, individual):
        """Bit flip mutation."""
        mutated = individual.copy()
        for i in range(len(mutated)):
            if random.random() < 0.1:  # 10% chance per bit
                mutated[i] = 1 - mutated[i]
        return mutated
    
    def is_valid(self, individual):
        """Check if individual is valid."""
        return len(individual) == self.string_length and all(bit in [0, 1] for bit in individual)


# Example Problem: Traveling Salesman Problem
class TSPGeneticProblem(GeneticProblem):
    """Traveling Salesman Problem for genetic algorithm."""
    
    def __init__(self, cities: List[Tuple[float, float]]):
        self.cities = cities
        self.n_cities = len(cities)
    
    def generate_individual(self):
        """Generate random tour."""
        tour = list(range(self.n_cities))
        random.shuffle(tour)
        return tour
    
    def calculate_fitness(self, individual):
        """Calculate tour distance (negative for minimization)."""
        total_distance = 0
        for i in range(self.n_cities):
            city1 = self.cities[individual[i]]
            city2 = self.cities[individual[(i + 1) % self.n_cities]]
            distance = math.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)
            total_distance += distance
        return -total_distance  # Negative for maximization
    
    def crossover(self, parent1, parent2):
        """Order crossover (OX)."""
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        
        # Create child1
        child1 = [-1] * size
        child1[start:end] = parent1[start:end]
        
        # Fill remaining positions from parent2
        remaining = [x for x in parent2 if x not in child1[start:end]]
        j = 0
        for i in range(size):
            if child1[i] == -1:
                child1[i] = remaining[j]
                j += 1
        
        # Create child2
        child2 = [-1] * size
        child2[start:end] = parent2[start:end]
        
        # Fill remaining positions from parent1
        remaining = [x for x in parent1 if x not in child2[start:end]]
        j = 0
        for i in range(size):
            if child2[i] == -1:
                child2[i] = remaining[j]
                j += 1
        
        return child1, child2
    
    def mutate(self, individual):
        """Swap mutation."""
        mutated = individual.copy()
        i, j = random.sample(range(len(mutated)), 2)
        mutated[i], mutated[j] = mutated[j], mutated[i]
        return mutated
    
    def is_valid(self, individual):
        """Check if tour is valid."""
        return len(set(individual)) == self.n_cities and len(individual) == self.n_cities


# Example Problem: Function Optimization
class FunctionOptimizationProblem(GeneticProblem):
    """Function optimization problem."""
    
    def __init__(self, objective_function: Callable, bounds: List[Tuple[float, float]]):
        self.objective_function = objective_function
        self.bounds = bounds
        self.dimension = len(bounds)
    
    def generate_individual(self):
        """Generate random real-valued individual."""
        individual = []
        for lower, upper in self.bounds:
            individual.append(random.uniform(lower, upper))
        return individual
    
    def calculate_fitness(self, individual):
        """Calculate function value."""
        return -self.objective_function(individual)  # Negative for maximization
    
    def crossover(self, parent1, parent2):
        """Arithmetic crossover."""
        alpha = random.random()
        child1 = [alpha * p1 + (1 - alpha) * p2 for p1, p2 in zip(parent1, parent2)]
        child2 = [(1 - alpha) * p1 + alpha * p2 for p1, p2 in zip(parent1, parent2)]
        return child1, child2
    
    def mutate(self, individual):
        """Gaussian mutation."""
        mutated = individual.copy()
        for i in range(len(mutated)):
            if random.random() < 0.1:  # 10% chance per dimension
                lower, upper = self.bounds[i]
                sigma = (upper - lower) * 0.1
                mutated[i] += random.gauss(0, sigma)
                mutated[i] = max(lower, min(upper, mutated[i]))  # Clamp to bounds
        return mutated
    
    def is_valid(self, individual):
        """Check if individual is within bounds."""
        for i, (lower, upper) in enumerate(self.bounds):
            if not (lower <= individual[i] <= upper):
                return False
        return True


def main():
    """Example usage of genetic algorithm."""
    print("=== Genetic Algorithm Framework Demo ===\n")
    
    # Example 1: Binary String Optimization
    print("1. Binary String Optimization:")
    print("Target: 10 ones in 20-bit string")
    
    binary_problem = BinaryStringProblem(string_length=20, target_ones=10)
    ga = GeneticAlgorithm(
        population_size=50,
        generations=100,
        mutation_rate=0.05,
        crossover_rate=0.8,
        verbose=True
    )
    
    result = ga.evolve(binary_problem)
    ones_count = sum(result.best_individual.genotype)
    print(f"Best solution: {result.best_individual.genotype}")
    print(f"Number of ones: {ones_count}")
    print(f"Fitness: {result.best_fitness}")
    
    print("\n" + "="*50 + "\n")
    
    # Example 2: Traveling Salesman Problem
    print("2. Traveling Salesman Problem:")
    cities = [
        (0, 0), (1, 2), (3, 1), (4, 3), (5, 0),
        (6, 2), (7, 4), (8, 1), (9, 3), (10, 0)
    ]
    
    tsp_problem = TSPGeneticProblem(cities)
    ga = GeneticAlgorithm(
        population_size=100,
        generations=200,
        mutation_rate=0.02,
        crossover_rate=0.9,
        verbose=True
    )
    
    result = ga.evolve(tsp_problem)
    tour_distance = -result.best_fitness  # Convert back to distance
    print(f"Best tour: {result.best_individual.genotype}")
    print(f"Tour distance: {tour_distance:.2f}")
    
    print("\n" + "="*50 + "\n")
    
    # Example 3: Function Optimization
    print("3. Function Optimization (Sphere Function):")
    
    def sphere_function(x):
        """Sphere function for testing optimization."""
        return sum(xi**2 for xi in x)
    
    bounds = [(-5.12, 5.12), (-5.12, 5.12)]  # 2D optimization
    func_problem = FunctionOptimizationProblem(sphere_function, bounds)
    
    ga = GeneticAlgorithm(
        population_size=80,
        generations=150,
        mutation_rate=0.03,
        crossover_rate=0.85,
        verbose=True
    )
    
    result = ga.evolve(func_problem)
    function_value = -result.best_fitness  # Convert back to function value
    print(f"Best solution: {result.best_individual.genotype}")
    print(f"Function value: {function_value:.6f}")


if __name__ == "__main__":
    import math
    main() 