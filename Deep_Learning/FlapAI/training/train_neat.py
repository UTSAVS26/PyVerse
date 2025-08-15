import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any, Tuple
import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.flappy_bird import FlappyBirdGame
from agents.neat_agent import NEATPopulation, NEATAgent

class NEATTrainer:
    """Trainer for NEAT agents."""
    
    def __init__(self, max_generations: int = 50, max_fitness: float = 1000,
                 render_training: bool = False, save_interval: int = 10):
        self.max_generations = max_generations
        self.max_fitness = max_fitness
        self.render_training = render_training
        self.save_interval = save_interval
        
        # Training statistics
        self.generation_stats = []
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.best_score_history = []
        
        # Create game and population
        self.game = FlappyBirdGame(headless=not render_training)
        self.population = NEATPopulation()
        
        # Best agent tracking
        self.best_agent = None
        self.best_score = 0
        
    def train(self) -> NEATAgent:
        """Train the NEAT population."""
        print(f"Starting NEAT training for {self.max_generations} generations...")
        print(f"Population size: {self.population.config.pop_size}")
        print(f"Fitness threshold: {self.max_fitness}")
        print("-" * 50)
        
        start_time = time.time()
        
        for generation in range(self.max_generations):
            print(f"\nGeneration {generation + 1}/{self.max_generations}")
            
            # Get agents for this generation
            agents = self.population.get_agents()
            
            # Evaluate all agents
            best_gen_fitness = float('-inf')
            best_gen_score = 0
            fitnesses = []
            scores = []
            
            for i, agent in enumerate(agents):
                # Reset agent fitness
                agent.reset_fitness()
                
                # Run episode
                score, fitness = self._evaluate_agent(agent)
                
                fitnesses.append(fitness)
                scores.append(score)
                
                # Track best in generation
                if fitness > best_gen_fitness:
                    best_gen_fitness = fitness
                    best_gen_score = score
                    
                # Update best overall
                if score > self.best_score:
                    self.best_score = score
                    self.best_agent = agent
                    
                # Progress indicator
                if (i + 1) % 10 == 0:
                    print(f"  Evaluated {i + 1}/{len(agents)} agents")
                    
            # Update population statistics
            avg_fitness = np.mean(fitnesses)
            avg_score = np.mean(scores)
            
            self.best_fitness_history.append(best_gen_fitness)
            self.avg_fitness_history.append(avg_fitness)
            self.best_score_history.append(best_gen_score)
            
            # Print generation results
            print(f"  Best Fitness: {best_gen_fitness:.2f}")
            print(f"  Avg Fitness: {avg_fitness:.2f}")
            print(f"  Best Score: {best_gen_score}")
            print(f"  Avg Score: {avg_score:.1f}")
            print(f"  Best Overall Score: {self.best_score}")
            
            # Check for convergence
            if best_gen_fitness >= self.max_fitness:
                print(f"\nFitness threshold reached! Best fitness: {best_gen_fitness}")
                break
                
            # Save best agent periodically
            if (generation + 1) % self.save_interval == 0:
                self._save_best_agent(generation + 1)
                
            # Advance to next generation
            if generation < self.max_generations - 1:
                agents = self.population.next_generation()
                
        # Training complete
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.1f} seconds")
        print(f"Best score achieved: {self.best_score}")
        
        # Save final best agent
        self._save_best_agent("final")
        
        # Plot training progress
        self._plot_training_progress()
        
        return self.best_agent
        
    def _evaluate_agent(self, agent: NEATAgent) -> Tuple[int, float]:
        """
        Evaluate a single agent.
        
        Args:
            agent: NEAT agent to evaluate
            
        Returns:
            (score, fitness): Final score and fitness
        """
        # Reset game
        state = self.game.reset()
        
        # Run episode
        done = False
        while not done:
            # Get action from agent
            action = agent.get_action(state)
            
            # Take step in game
            next_state, reward, done, info = self.game.step(action)
            
            # Update agent
            agent.update(state, action, reward, next_state, done)
            
            # Render if requested
            if self.render_training:
                self.game.render()
                
            state = next_state
            
        # Return final score and fitness
        score = info.get('score', 0)
        fitness = agent.get_fitness()
        
        return score, fitness
        
    def _save_best_agent(self, generation: str) -> None:
        """Save the best agent to file."""
        if self.best_agent:
            filename = f"best_neat_gen_{generation}.pkl"
            filepath = os.path.join("models", filename)
            
            # Create models directory if it doesn't exist
            os.makedirs("models", exist_ok=True)
            
            self.best_agent.save(filepath)
            print(f"  Saved best agent to {filepath}")
            
    def _plot_training_progress(self) -> None:
        """Plot training progress."""
        if not self.generation_stats:
            return
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Fitness plot
        generations = range(1, len(self.best_fitness_history) + 1)
        ax1.plot(generations, self.best_fitness_history, 'b-', label='Best Fitness')
        ax1.plot(generations, self.avg_fitness_history, 'r-', label='Avg Fitness')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Fitness')
        ax1.set_title('NEAT Training Progress - Fitness')
        ax1.legend()
        ax1.grid(True)
        
        # Score plot
        ax2.plot(generations, self.best_score_history, 'g-', label='Best Score')
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Score')
        ax2.set_title('NEAT Training Progress - Score')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('neat_training_progress.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        return {
            'generations_completed': len(self.best_fitness_history),
            'best_fitness': max(self.best_fitness_history) if self.best_fitness_history else 0,
            'best_score': self.best_score,
            'final_avg_fitness': self.avg_fitness_history[-1] if self.avg_fitness_history else 0,
            'fitness_history': self.best_fitness_history,
            'avg_fitness_history': self.avg_fitness_history,
            'score_history': self.best_score_history
        }

def train_neat(generations: int = 50, population_size: int = 50, 
               render: bool = False, save_interval: int = 10) -> NEATAgent:
    """
    Train a NEAT agent.
    
    Args:
        generations: Number of generations to train
        population_size: Size of the population
        render: Whether to render training
        save_interval: How often to save the best agent
        
    Returns:
        best_agent: The best trained agent
    """
    trainer = NEATTrainer(
        max_generations=generations,
        render_training=render,
        save_interval=save_interval
    )
    
    return trainer.train()

def main():
    """Main function for NEAT training."""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Train NEAT agent for FlapAI')
    parser.add_argument('--generations', type=int, default=50, 
                       help='Number of generations to train')
    parser.add_argument('--population', type=int, default=50,
                       help='Population size')
    parser.add_argument('--render', action='store_true',
                       help='Render training')
    parser.add_argument('--save-interval', type=int, default=10,
                       help='Save interval for best agent')
    
    args = parser.parse_args()
    
    # Train the agent
    best_agent = train_neat(
        generations=args.generations,
        population_size=args.population,
        render=args.render,
        save_interval=args.save_interval
    )
    
    print(f"\nTraining complete! Best agent saved.")
    print(f"Best score achieved: {best_agent.best_score}")

if __name__ == "__main__":
    main() 