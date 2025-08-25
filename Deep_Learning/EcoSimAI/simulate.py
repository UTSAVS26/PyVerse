import pygame
import matplotlib.pyplot as plt
import numpy as np
import time
import argparse
from typing import List, Dict
from environment import Environment, CellType
from agents import Prey, Predator, Plant
from rl_agent import RLPrey, RLPredator

class SimulationVisualizer:
    """Visualization class for the ecosystem simulation."""
    
    def __init__(self, environment: Environment, cell_size: int = 10, 
                 use_pygame: bool = True, show_stats: bool = True):
        self.environment = environment
        self.cell_size = cell_size
        self.use_pygame = use_pygame
        self.show_stats = show_stats
        
        # Color mapping
        self.colors = {
            CellType.EMPTY.value: (255, 255, 255),  # White
            CellType.PLANT.value: (34, 139, 34),    # Forest Green
            CellType.PREY.value: (255, 215, 0),     # Gold
            CellType.PREDATOR.value: (220, 20, 60)  # Crimson
        }
        
        # Statistics tracking
        self.stats_history = {
            'plants': [],
            'prey': [],
            'predators': [],
            'total_agents': []
        }
        
        if self.use_pygame:
            self._init_pygame()
        else:
            self._init_matplotlib()
    
    def _init_pygame(self):
        """Initialize pygame display."""
        pygame.init()
        
        # Calculate window size
        self.width = self.environment.width * self.cell_size
        self.height = self.environment.height * self.cell_size
        
        if self.show_stats:
            self.height += 100  # Extra space for stats
        
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("EcoSimAI - Virtual Ecosystem Simulation")
        self.font = pygame.font.Font(None, 24)
    
    def _init_matplotlib(self):
        """Initialize matplotlib for visualization."""
        plt.ion()  # Turn on interactive mode
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 5))
        self.fig.suptitle('EcoSimAI - Virtual Ecosystem Simulation')
    
    def update(self, step: int):
        """Update the visualization."""
        if self.use_pygame:
            self._update_pygame(step)
        else:
            self._update_matplotlib(step)
    
    def _update_pygame(self, step: int):
        """Update pygame display."""
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        
        # Clear screen
        self.screen.fill((255, 255, 255))
        
        # Draw grid
        grid = self.environment.get_grid_state()
        for y in range(self.environment.height):
            for x in range(self.environment.width):
                cell_value = grid[y, x]
                color = self.colors[cell_value]
                
                rect = pygame.Rect(
                    x * self.cell_size, 
                    y * self.cell_size, 
                    self.cell_size, 
                    self.cell_size
                )
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)
        
        # Draw statistics
        if self.show_stats:
            stats = self.environment.get_statistics()
            self._draw_stats_pygame(stats, step)
        
        pygame.display.flip()
        return True
    
    def _draw_stats_pygame(self, stats: Dict, step: int):
        """Draw statistics on pygame screen."""
        y_offset = self.environment.height * self.cell_size + 10
        
        # Step counter
        step_text = self.font.render(f"Step: {step}", True, (0, 0, 0))
        self.screen.blit(step_text, (10, y_offset))
        
        # Population counts
        plants_text = self.font.render(f"Plants: {stats['plants']}", True, (34, 139, 34))
        self.screen.blit(plants_text, (10, y_offset + 25))
        
        prey_text = self.font.render(f"Prey: {stats['prey']}", True, (255, 215, 0))
        self.screen.blit(prey_text, (10, y_offset + 50))
        
        predator_text = self.font.render(f"Predators: {stats['predators']}", True, (220, 20, 60))
        self.screen.blit(predator_text, (10, y_offset + 75))
    
    def _update_matplotlib(self, step: int):
        """Update matplotlib display."""
        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()
        
        # Draw grid
        grid = self.environment.get_grid_state()
        self.ax1.imshow(grid, cmap='tab20', interpolation='nearest')
        self.ax1.set_title(f'Ecosystem Grid - Step {step}')
        self.ax1.set_xlabel('X')
        self.ax1.set_ylabel('Y')
        
        # Update statistics
        stats = self.environment.get_statistics()
        for key in self.stats_history:
            if key in stats:
                self.stats_history[key].append(stats[key])
        
        # Plot population trends
        steps = list(range(len(self.stats_history['plants'])))
        if steps:
            self.ax2.plot(steps, self.stats_history['plants'], 'g-', label='Plants', linewidth=2)
            self.ax2.plot(steps, self.stats_history['prey'], 'y-', label='Prey', linewidth=2)
            self.ax2.plot(steps, self.stats_history['predators'], 'r-', label='Predators', linewidth=2)
            self.ax2.set_title('Population Trends')
            self.ax2.set_xlabel('Step')
            self.ax2.set_ylabel('Count')
            self.ax2.legend()
            self.ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.pause(0.01)
    
    def close(self):
        """Close the visualization."""
        if self.use_pygame:
            pygame.quit()
        else:
            plt.ioff()
            plt.show()

def create_environment_with_rl_agents(width: int = 50, height: int = 50,
                                    rl_prey_ratio: float = 0.3, rl_predator_ratio: float = 0.3):
    """Create environment with mix of regular and RL agents."""
    env = Environment(width, height)
    
    # Clear existing agents
    env.agents.clear()
    env.agent_positions.clear()
    env.grid.fill(CellType.EMPTY.value)
    
    # Add plants
    for _ in range(100):
        pos = env._get_random_empty_position()
        if pos:
            env.grid[pos.y, pos.x] = CellType.PLANT.value
    
    # Add prey (mix of regular and RL)
    for i in range(50):
        pos = env._get_random_empty_position()
        if pos:
            if random.random() < rl_prey_ratio:
                prey = RLPrey(env._get_next_agent_id(), pos, energy=50)
            else:
                prey = Prey(env._get_next_agent_id(), pos, energy=50)
            env._add_agent(prey, pos)
    
    # Add predators (mix of regular and RL)
    for i in range(20):
        pos = env._get_random_empty_position()
        if pos:
            if random.random() < rl_predator_ratio:
                predator = RLPredator(env._get_next_agent_id(), pos, energy=50)
            else:
                predator = Predator(env._get_next_agent_id(), pos, energy=50)
            env._add_agent(predator, pos)
    
    return env

def run_simulation(env: Environment, max_steps: int = 1000, 
                  visualization_speed: float = 0.1,
                  use_pygame: bool = True,
                  save_stats: bool = False):
    """Run the ecosystem simulation."""
    
    # Initialize visualizer
    visualizer = SimulationVisualizer(env, cell_size=8, use_pygame=use_pygame)
    
    # Statistics tracking
    all_stats = []
    
    print(f"Starting simulation for {max_steps} steps...")
    print("Controls: Close window to stop simulation")
    
    try:
        for step in range(max_steps):
            # Update environment
            env.step()
            
            # Get statistics
            stats = env.get_statistics()
            all_stats.append(stats)
            
            # Update visualization
            if not visualizer.update(step):
                break
            
            # Print progress
            if step % 100 == 0:
                print(f"Step {step}: Plants={stats['plants']}, "
                      f"Prey={stats['prey']}, Predators={stats['predators']}")
            
            # Check for extinction
            if stats['prey'] == 0 and stats['predators'] == 0:
                print(f"Extinction event at step {step}!")
                break
            
            # Control simulation speed
            time.sleep(visualization_speed)
    
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
    
    finally:
        visualizer.close()
        
        # Save statistics if requested
        if save_stats:
            save_simulation_stats(all_stats, "simulation_stats.csv")
        
        # Print final statistics
        if all_stats:
            final_stats = all_stats[-1]
            print(f"\nFinal Statistics:")
            print(f"Total Steps: {len(all_stats)}")
            print(f"Final Plants: {final_stats['plants']}")
            print(f"Final Prey: {final_stats['prey']}")
            print(f"Final Predators: {final_stats['predators']}")
            print(f"Total Agents: {final_stats['total_agents']}")

def save_simulation_stats(stats_list: List[Dict], filename: str):
    """Save simulation statistics to CSV file."""
    import csv
    
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['step', 'plants', 'prey', 'predators', 'total_agents']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for stats in stats_list:
            writer.writerow(stats)
    
    print(f"Statistics saved to {filename}")

def run_headless_simulation(env: Environment, max_steps: int = 1000, 
                           print_interval: int = 100):
    """Run simulation without visualization for performance testing."""
    
    print(f"Running headless simulation for {max_steps} steps...")
    
    all_stats = []
    
    for step in range(max_steps):
        env.step()
        stats = env.get_statistics()
        all_stats.append(stats)
        
        if step % print_interval == 0:
            print(f"Step {step}: Plants={stats['plants']}, "
                  f"Prey={stats['prey']}, Predators={stats['predators']}")
        
        # Check for extinction
        if stats['prey'] == 0 and stats['predators'] == 0:
            print(f"Extinction event at step {step}!")
            break
    
    # Print final statistics
    if all_stats:
        final_stats = all_stats[-1]
        print(f"\nFinal Statistics:")
        print(f"Total Steps: {len(all_stats)}")
        print(f"Final Plants: {final_stats['plants']}")
        print(f"Final Prey: {final_stats['prey']}")
        print(f"Final Predators: {final_stats['predators']}")
        print(f"Total Agents: {final_stats['total_agents']}")
    
    return all_stats

def main():
    """Main function to run the simulation."""
    parser = argparse.ArgumentParser(description='EcoSimAI - Virtual Ecosystem Simulation')
    parser.add_argument('--steps', type=int, default=1000, help='Number of simulation steps')
    parser.add_argument('--width', type=int, default=50, help='Environment width')
    parser.add_argument('--height', type=int, default=50, help='Environment height')
    parser.add_argument('--speed', type=float, default=0.1, help='Visualization speed (seconds per step)')
    parser.add_argument('--headless', action='store_true', help='Run without visualization')
    parser.add_argument('--matplotlib', action='store_true', help='Use matplotlib instead of pygame')
    parser.add_argument('--rl-prey', type=float, default=0.3, help='Ratio of RL prey agents')
    parser.add_argument('--rl-predators', type=float, default=0.3, help='Ratio of RL predator agents')
    parser.add_argument('--save-stats', action='store_true', help='Save statistics to CSV')
    
    args = parser.parse_args()
    
    # Create environment
    if args.rl_prey > 0 or args.rl_predators > 0:
        env = create_environment_with_rl_agents(
            args.width, args.height, args.rl_prey, args.rl_predators
        )
    else:
        env = Environment(args.width, args.height)
    
    # Run simulation
    if args.headless:
        run_headless_simulation(env, args.steps)
    else:
        run_simulation(
            env, 
            max_steps=args.steps,
            visualization_speed=args.speed,
            use_pygame=not args.matplotlib,
            save_stats=args.save_stats
        )

if __name__ == "__main__":
    import random
    main()
