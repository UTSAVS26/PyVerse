import neat
import numpy as np
import pickle
from typing import Dict, Any, List, Tuple
import os
from .base_agent import BaseAgent

class NEATAgent(BaseAgent):
    """NEAT (NeuroEvolution of Augmenting Topologies) agent for FlapAI."""
    
    def __init__(self, config_file: str = None, genome: neat.DefaultGenome = None):
        super().__init__("NEATAgent")
        
        if config_file is None:
            config_file = os.path.join(os.path.dirname(__file__), '..', 'config', 'neat-config.txt')
            
        # Load NEAT configuration
        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                 neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)
        
        # Create or use provided genome
        if genome is None:
            self.genome = neat.DefaultGenome(1)
            self.genome.configure_new(self.config.genome_config)
        else:
            self.genome = genome
            
        # Create neural network
        self.net = neat.nn.FeedForwardNetwork.create(self.genome, self.config)
        
        # Training variables
        self.fitness = 0
        self.current_score = 0
        
    def get_action(self, state: Dict[str, Any]) -> int:
        """
        Get action from NEAT neural network.
        
        Args:
            state: Current game state
            
        Returns:
            action: 0 for no flap, 1 for flap
        """
        # Convert state to input vector
        inputs = self._state_to_inputs(state)
        
        # Get network output
        output = self.net.activate(inputs)
        
        # Convert output to action (threshold at 0.5)
        action = 1 if output[0] > 0.5 else 0
        
        return action
        
    def update(self, state: Dict[str, Any], action: int, 
               reward: float, next_state: Dict[str, Any], done: bool) -> None:
        """
        Update fitness for NEAT training.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        # Update fitness based on reward
        self.fitness += reward
        
        # Update current score
        if 'score' in state:
            self.current_score = state['score']
            
        # If episode is done, finalize fitness
        if done:
            # Bonus for survival time
            survival_bonus = state.get('frame_count', 0) * 0.1
            
            # Bonus for score
            score_bonus = self.current_score * 10
            
            # Penalty for dying early
            if self.current_score == 0:
                self.fitness -= 50
                
            self.fitness += survival_bonus + score_bonus
            
    def _state_to_inputs(self, state: Dict[str, Any]) -> List[float]:
        """
        Convert game state to neural network inputs.
        
        Args:
            state: Game state dictionary
            
        Returns:
            inputs: List of float inputs for neural network
        """
        inputs = [
            state.get('bird_y', 0.5),
            state.get('bird_velocity', 0.0),
            state.get('pipe_x', 1.0),
            state.get('pipe_gap_y', 0.5),
            state.get('pipe_gap_size', 0.25),
            state.get('distance_to_pipe', 1.0),
            state.get('bird_alive', 1.0)
        ]
        
        return inputs
        
    def get_fitness(self) -> float:
        """Get current fitness value."""
        return self.fitness
        
    def reset_fitness(self) -> None:
        """Reset fitness for new generation."""
        self.fitness = 0
        self.current_score = 0
        
    def save(self, filepath: str) -> None:
        """Save the genome to a file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.genome, f)
            
    def load(self, filepath: str) -> None:
        """Load the genome from a file."""
        with open(filepath, 'rb') as f:
            self.genome = pickle.load(f)
            
        # Recreate neural network
        self.net = neat.nn.FeedForwardNetwork.create(self.genome, self.config)
        
    def get_stats(self) -> Dict[str, Any]:
        """Get current agent statistics."""
        stats = super().get_stats()
        stats.update({
            'fitness': self.fitness,
            'current_score': self.current_score,
            'genome_id': self.genome.key if hasattr(self.genome, 'key') else 'unknown'
        })
        return stats

class NEATPopulation:
    """Manages a population of NEAT agents."""
    
    def __init__(self, config_file: str = None):
        if config_file is None:
            config_file = os.path.join(os.path.dirname(__file__), '..', 'config', 'neat-config.txt')
            
        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                 neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)
        self.population = neat.Population(self.config)
        
        # Statistics
        self.generation = 0
        self.best_fitness = 0
        self.avg_fitness = 0
        self.best_genome = None
        
    def get_agents(self) -> List[NEATAgent]:
        """Get all agents in the current population."""
        agents = []
        for genome_id, genome in self.population.population.items():
            agent = NEATAgent(config_file=None, genome=genome)
            agents.append(agent)
        return agents
        
    def get_best_agent(self) -> NEATAgent:
        """Get the best agent from the current population."""
        if self.best_genome is None:
            # Find best genome
            best_genome = None
            best_fitness = float('-inf')
            
            for genome_id, genome in self.population.population.items():
                if genome.fitness > best_fitness:
                    best_fitness = genome.fitness
                    best_genome = genome
                    
            self.best_genome = best_genome
            
        return NEATAgent(config_file=None, genome=self.best_genome)
        
    def next_generation(self) -> List[NEATAgent]:
        """Advance to the next generation and return new agents."""
        self.generation += 1
        
        # Define a simple fitness function that assigns random fitness values
        def fitness_function(genomes, config):
            for genome_id, genome in genomes:
                genome.fitness = 0.0  # Default fitness
        
        # Run NEAT evolution
        self.population.run(fitness_function, 1)
        
        # Update statistics
        fitnesses = [genome.fitness for genome in self.population.population.values() if genome.fitness is not None]
        if fitnesses:
            self.avg_fitness = np.mean(fitnesses)
            self.best_fitness = max(fitnesses)
        else:
            self.avg_fitness = 0
            self.best_fitness = 0
        
        # Get new agents
        return self.get_agents()
        
    def get_stats(self) -> Dict[str, Any]:
        """Get population statistics."""
        return {
            'generation': self.generation,
            'population_size': len(self.population.population),
            'best_fitness': self.best_fitness,
            'avg_fitness': self.avg_fitness,
            'species_count': len(self.population.species.species)
        }
        
    def save_best(self, filepath: str) -> None:
        """Save the best genome."""
        if self.best_genome:
            with open(filepath, 'wb') as f:
                pickle.dump(self.best_genome, f)
                
    def load_best(self, filepath: str) -> NEATAgent:
        """Load the best genome from file."""
        with open(filepath, 'rb') as f:
            genome = pickle.load(f)
            
        self.best_genome = genome
        return NEATAgent(config_file=None, genome=genome) 