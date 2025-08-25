"""
MazeMind - Simulation and Visualization Module

This module provides simulation, visualization, and comparison tools for the MazeMind system.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import time
from typing import Tuple, List, Dict, Optional
import random

from maze_generator import MazeGenerator
from pathfinding import PathFinder
from rl_agent import QLearningAgent, MultiAgentSystem


class MazeSimulator:
    """Main simulation class for maze generation, solving, and visualization."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize simulator.
        
        Args:
            figsize: Figure size for visualizations
        """
        self.figsize = figsize
        self.pathfinder = PathFinder()
        self.generator = MazeGenerator()
        
    def generate_and_solve(self, width: int = 20, height: int = 20, 
                          algorithm: str = "a_star", 
                          generation_method: str = "dfs") -> Dict:
        """
        Generate a maze and solve it with specified algorithm.
        
        Args:
            width: Maze width
            height: Maze height
            algorithm: Pathfinding algorithm ("bfs", "dfs", "a_star", "dijkstra")
            generation_method: Maze generation method ("dfs", "prims", "recursive_division")
            
        Returns:
            Results dictionary
        """
        # Generate maze
        self.generator = MazeGenerator(width, height)
        
        if generation_method == "dfs":
            maze = self.generator.generate_dfs()
        elif generation_method == "prims":
            maze = self.generator.generate_prims()
        elif generation_method == "recursive_division":
            maze = self.generator.generate_recursive_division()
        else:
            raise ValueError(f"Unknown generation method: {generation_method}")
        
        # Get start and end points
        start, goal = self.generator.get_start_end_points()
        
        # Solve maze
        result = self.pathfinder.solve_maze(maze, start, goal, algorithm)
        result['maze'] = maze
        result['start'] = start
        result['goal'] = goal
        result['generation_method'] = generation_method
        
        return result
    
    def visualize_path(self, maze: np.ndarray, path: List[Tuple[int, int]], 
                      start: Tuple[int, int], goal: Tuple[int, int], 
                      title: str = "Maze Solution", save_path: Optional[str] = None):
        """
        Visualize maze with solution path.
        
        Args:
            maze: Maze grid
            path: Solution path
            start: Starting position
            goal: Goal position
            title: Plot title
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot maze
        height, width = maze.shape
        ax.imshow(maze, cmap='binary', origin='upper')
        
        # Plot path
        if path:
            path_x = [pos[0] for pos in path]
            path_y = [pos[1] for pos in path]
            ax.plot(path_x, path_y, 'r-', linewidth=2, label='Solution Path')
            ax.plot(path_x, path_y, 'ro', markersize=4)
        
        # Plot start and goal
        ax.plot(start[0], start[1], 'go', markersize=10, label='Start')
        ax.plot(goal[0], goal[1], 'bo', markersize=10, label='Goal')
        
        # Customize plot
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set axis limits
        ax.set_xlim(-0.5, width - 0.5)
        ax.set_ylim(height - 0.5, -0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def animate_solution(self, maze: np.ndarray, path: List[Tuple[int, int]], 
                        start: Tuple[int, int], goal: Tuple[int, int], 
                        interval: int = 200, save_path: Optional[str] = None):
        """
        Create animated visualization of pathfinding.
        
        Args:
            maze: Maze grid
            path: Solution path
            start: Starting position
            goal: Goal position
            interval: Animation interval (ms)
            save_path: Path to save animation
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot maze
        height, width = maze.shape
        ax.imshow(maze, cmap='binary', origin='upper')
        
        # Plot start and goal
        ax.plot(start[0], start[1], 'go', markersize=10, label='Start')
        ax.plot(goal[0], goal[1], 'bo', markersize=10, label='Goal')
        
        # Initialize path line
        line, = ax.plot([], [], 'r-', linewidth=2, label='Solution Path')
        point, = ax.plot([], [], 'ro', markersize=6)
        
        ax.set_title('Maze Solution Animation', fontsize=16, fontweight='bold')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.5, width - 0.5)
        ax.set_ylim(height - 0.5, -0.5)
        
        def animate(frame):
            if frame < len(path):
                # Update path line
                path_x = [pos[0] for pos in path[:frame+1]]
                path_y = [pos[1] for pos in path[:frame+1]]
                line.set_data(path_x, path_y)
                
                # Update current position
                point.set_data([path[frame][0]], [path[frame][1]])
            
            return line, point
        
        anim = animation.FuncAnimation(fig, animate, frames=len(path), 
                                     interval=interval, blit=True, repeat=True)
        
        plt.tight_layout()
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=5)
        
        plt.show()
        return anim
    
    def train_episode(self, agent: QLearningAgent, maze: np.ndarray, 
                     start: Tuple[int, int], goal: Tuple[int, int]) -> Tuple[bool, int]:
        """
        Train RL agent for one episode.
        
        Args:
            agent: Q-learning agent
            maze: Maze grid
            start: Starting position
            goal: Goal position
            
        Returns:
            Tuple of (success, steps)
        """
        result = agent.train_episode(maze, start, goal)
        return result['success'], result['steps']
    
    def train_agent(self, agent: QLearningAgent, episodes: int = 1000, 
                   maze_size: int = 20) -> Dict:
        """
        Train RL agent for multiple episodes.
        
        Args:
            agent: Q-learning agent
            episodes: Number of training episodes
            maze_size: Size of training mazes
            
        Returns:
            Training results
        """
        print(f"Training agent for {episodes} episodes...")
        
        success_count = 0
        total_steps = 0
        training_history = []
        
        for episode in range(episodes):
            # Generate new maze for each episode
            generator = MazeGenerator(maze_size, maze_size)
            maze = generator.generate_dfs()
            start, goal = generator.get_start_end_points()
            
            # Train episode
            success, steps = self.train_episode(agent, maze, start, goal)
            
            if success:
                success_count += 1
            total_steps += steps
            
            training_history.append({
                'episode': episode,
                'success': success,
                'steps': steps
            })
            
            # Print progress
            if episode % 100 == 0:
                success_rate = success_count / (episode + 1)
                avg_steps = total_steps / (episode + 1)
                print(f"Episode {episode}: Success Rate={success_rate:.3f}, "
                      f"Avg Steps={avg_steps:.1f}")
        
        final_success_rate = success_count / episodes
        final_avg_steps = total_steps / episodes
        
        return {
            'episodes': episodes,
            'final_success_rate': final_success_rate,
            'final_avg_steps': final_avg_steps,
            'training_history': training_history,
            'agent_stats': agent.get_training_stats()
        }
    
    def compare_algorithms(self, maze_sizes: List[int] = [10, 20, 30], 
                          num_trials: int = 10) -> Dict:
        """
        Compare different pathfinding algorithms.
        
        Args:
            maze_sizes: List of maze sizes to test
            num_trials: Number of trials per size
            
        Returns:
            Comparison results
        """
        algorithms = ['bfs', 'dfs', 'a_star', 'dijkstra']
        results = {size: {alg: [] for alg in algorithms} for size in maze_sizes}
        
        for size in maze_sizes:
            print(f"Testing maze size: {size}x{size}")
            
            for trial in range(num_trials):
                # Generate maze
                generator = MazeGenerator(size, size)
                maze = generator.generate_dfs()
                start, goal = generator.get_start_end_points()
                
                # Test each algorithm
                for algorithm in algorithms:
                    result = self.pathfinder.solve_maze(maze, start, goal, algorithm)
                    results[size][algorithm].append(result)
        
        return results
    
    def plot_comparison(self, results: Dict):
        """
        Plot algorithm comparison results.
        
        Args:
            results: Results from compare_algorithms
        """
        algorithms = ['bfs', 'dfs', 'a_star', 'dijkstra']
        maze_sizes = list(results.keys())
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Success rate
        for algorithm in algorithms:
            success_rates = []
            for size in maze_sizes:
                trials = results[size][algorithm]
                success_rate = sum(1 for r in trials if r['success']) / len(trials)
                success_rates.append(success_rate)
            ax1.plot(maze_sizes, success_rates, 'o-', label=algorithm.upper())
        
        ax1.set_title('Success Rate vs Maze Size')
        ax1.set_xlabel('Maze Size')
        ax1.set_ylabel('Success Rate')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Average path length
        for algorithm in algorithms:
            path_lengths = []
            for size in maze_sizes:
                trials = results[size][algorithm]
                successful_trials = [r for r in trials if r['success']]
                if successful_trials:
                    avg_length = np.mean([r['path_length'] for r in successful_trials])
                    path_lengths.append(avg_length)
                else:
                    path_lengths.append(0)
            ax2.plot(maze_sizes, path_lengths, 'o-', label=algorithm.upper())
        
        ax2.set_title('Average Path Length vs Maze Size')
        ax2.set_xlabel('Maze Size')
        ax2.set_ylabel('Path Length')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Average execution time
        for algorithm in algorithms:
            execution_times = []
            for size in maze_sizes:
                trials = results[size][algorithm]
                avg_time = np.mean([r['execution_time'] for r in trials])
                execution_times.append(avg_time * 1000)  # Convert to ms
            ax3.plot(maze_sizes, execution_times, 'o-', label=algorithm.upper())
        
        ax3.set_title('Average Execution Time vs Maze Size')
        ax3.set_xlabel('Maze Size')
        ax3.set_ylabel('Execution Time (ms)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Average nodes explored
        for algorithm in algorithms:
            nodes_explored = []
            for size in maze_sizes:
                trials = results[size][algorithm]
                avg_nodes = np.mean([r['nodes_explored'] for r in trials])
                nodes_explored.append(avg_nodes)
            ax4.plot(maze_sizes, nodes_explored, 'o-', label=algorithm.upper())
        
        ax4.set_title('Average Nodes Explored vs Maze Size')
        ax4.set_xlabel('Maze Size')
        ax4.set_ylabel('Nodes Explored')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_training_progress(self, training_results: Dict):
        """
        Plot RL agent training progress.
        
        Args:
            training_results: Results from train_agent
        """
        history = training_results['training_history']
        
        episodes = [h['episode'] for h in history]
        successes = [h['success'] for h in history]
        steps = [h['steps'] for h in history]
        
        # Calculate moving averages
        window = 50
        success_rate_ma = []
        steps_ma = []
        
        for i in range(len(episodes)):
            start_idx = max(0, i - window + 1)
            recent_successes = successes[start_idx:i+1]
            recent_steps = steps[start_idx:i+1]
            
            success_rate_ma.append(sum(recent_successes) / len(recent_successes))
            steps_ma.append(np.mean(recent_steps))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Success rate over time
        ax1.plot(episodes, success_rate_ma, 'b-', linewidth=2)
        ax1.set_title('Training Progress - Success Rate')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Success Rate (Moving Average)')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Steps over time
        ax2.plot(episodes, steps_ma, 'r-', linewidth=2)
        ax2.set_title('Training Progress - Average Steps')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Steps (Moving Average)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def interactive_demo(self):
        """
        Run interactive demonstration of the MazeMind system.
        """
        print("🌀 Welcome to MazeMind Interactive Demo!")
        print("=" * 50)
        
        while True:
            print("\nChoose an option:")
            print("1. Generate and solve maze with classical algorithms")
            print("2. Train RL agent")
            print("3. Compare algorithms")
            print("4. Visualize maze generation")
            print("5. Exit")
            
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == '1':
                self._demo_classical_algorithms()
            elif choice == '2':
                self._demo_rl_training()
            elif choice == '3':
                self._demo_algorithm_comparison()
            elif choice == '4':
                self._demo_maze_generation()
            elif choice == '5':
                print("Thanks for using MazeMind!")
                break
            else:
                print("Invalid choice. Please try again.")
    
    def _demo_classical_algorithms(self):
        """Demo classical pathfinding algorithms."""
        print("\n--- Classical Algorithm Demo ---")
        
        # Generate maze
        size = int(input("Enter maze size (10-50): ") or "20")
        maze = self.generator.generate_dfs()
        start, goal = self.generator.get_start_end_points()
        
        # Solve with different algorithms
        algorithms = ['bfs', 'dfs', 'a_star', 'dijkstra']
        
        for algorithm in algorithms:
            result = self.pathfinder.solve_maze(maze, start, goal, algorithm)
            print(f"\n{algorithm.upper()}:")
            print(f"  Success: {result['success']}")
            print(f"  Path Length: {result['path_length']}")
            print(f"  Execution Time: {result['execution_time']:.4f}s")
            print(f"  Nodes Explored: {result['nodes_explored']}")
        
        # Visualize best solution
        best_result = self.pathfinder.solve_maze(maze, start, goal, 'a_star')
        if best_result['success']:
            self.visualize_path(maze, best_result['path'], start, goal, 
                              "A* Solution")
    
    def _demo_rl_training(self):
        """Demo RL agent training."""
        print("\n--- RL Agent Training Demo ---")
        
        episodes = int(input("Enter number of training episodes (100-1000): ") or "500")
        
        # Create and train agent
        agent = QLearningAgent(learning_rate=0.1, epsilon=0.2)
        results = self.train_agent(agent, episodes, maze_size=15)
        
        print(f"\nTraining completed!")
        print(f"Final Success Rate: {results['final_success_rate']:.3f}")
        print(f"Final Average Steps: {results['final_avg_steps']:.1f}")
        
        # Plot training progress
        self.plot_training_progress(results)
        
        # Test on new maze
        print("\nTesting on new maze...")
        test_maze = self.generator.generate_dfs()
        start, goal = self.generator.get_start_end_points()
        test_result = agent.solve_maze(test_maze, start, goal)
        
        if test_result['success']:
            self.visualize_path(test_maze, test_result['path'], start, goal, 
                              "RL Agent Solution")
    
    def _demo_algorithm_comparison(self):
        """Demo algorithm comparison."""
        print("\n--- Algorithm Comparison Demo ---")
        
        results = self.compare_algorithms(maze_sizes=[10, 15, 20], num_trials=5)
        self.plot_comparison(results)
    
    def _demo_maze_generation(self):
        """Demo different maze generation methods."""
        print("\n--- Maze Generation Demo ---")
        
        size = int(input("Enter maze size (10-30): ") or "15")
        
        methods = ['dfs', 'prims', 'recursive_division']
        
        for method in methods:
            print(f"\nGenerating maze using {method}...")
            
            if method == 'dfs':
                maze = self.generator.generate_dfs()
            elif method == 'prims':
                maze = self.generator.generate_prims()
            else:
                maze = self.generator.generate_recursive_division()
            
            start, goal = self.generator.get_start_end_points()
            self.visualize_path(maze, [], start, goal, f"{method.upper()} Maze")


if __name__ == "__main__":
    # Run interactive demo
    simulator = MazeSimulator()
    simulator.interactive_demo()
