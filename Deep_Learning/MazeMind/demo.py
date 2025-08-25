#!/usr/bin/env python3
"""
MazeMind Demo Script

This script demonstrates the key features of the MazeMind system.
"""

import numpy as np
from maze_generator import MazeGenerator
from pathfinding import PathFinder
from rl_agent import QLearningAgent
from simulate import MazeSimulator


def demo_classical_algorithms():
    """Demonstrate classical pathfinding algorithms."""
    print("=" * 60)
    print("🌀 MAZEMIND - Classical Algorithm Demo")
    print("=" * 60)
    
    # Generate maze
    generator = MazeGenerator(15, 15)
    maze = generator.generate_dfs()
    start, goal = generator.get_start_end_points()
    
    print(f"Maze size: {maze.shape}")
    print(f"Start: {start}, Goal: {goal}")
    print()
    
    # Test different algorithms
    pathfinder = PathFinder()
    algorithms = ['bfs', 'dfs', 'a_star', 'dijkstra']
    
    print("Algorithm Comparison:")
    print("-" * 50)
    print(f"{'Algorithm':<12} {'Success':<8} {'Path Length':<12} {'Time (ms)':<10} {'Nodes Explored':<15}")
    print("-" * 50)
    
    for algorithm in algorithms:
        result = pathfinder.solve_maze(maze, start, goal, algorithm)
        success = "✓" if result['success'] else "✗"
        path_len = result['path_length'] if result['success'] else "N/A"
        time_ms = f"{result['execution_time']*1000:.2f}"
        nodes = result['nodes_explored']
        
        print(f"{algorithm.upper():<12} {success:<8} {path_len:<12} {time_ms:<10} {nodes:<15}")
    
    print()


def demo_rl_training():
    """Demonstrate RL agent training."""
    print("=" * 60)
    print("🌀 MAZEMIND - RL Agent Training Demo")
    print("=" * 60)
    
    # Create agent
    agent = QLearningAgent(
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon=0.2,
        epsilon_decay=0.995
    )
    
    print("Training Q-Learning agent...")
    print("Parameters:")
    print(f"  Learning Rate: {agent.learning_rate}")
    print(f"  Discount Factor: {agent.discount_factor}")
    print(f"  Initial Epsilon: {agent.epsilon}")
    print(f"  Epsilon Decay: {agent.epsilon_decay}")
    print()
    
    # Train agent
    simulator = MazeSimulator()
    results = simulator.train_agent(agent, episodes=100, maze_size=11)
    
    print("Training Results:")
    print(f"  Episodes: {results['episodes']}")
    print(f"  Final Success Rate: {results['final_success_rate']:.3f}")
    print(f"  Final Average Steps: {results['final_avg_steps']:.1f}")
    print()
    
    # Test on new maze
    print("Testing on new maze...")
    generator = MazeGenerator(11, 11)
    test_maze = generator.generate_dfs()
    start, goal = generator.get_start_end_points()
    
    test_result = agent.solve_maze(test_maze, start, goal)
    print(f"Test Success: {'✓' if test_result['success'] else '✗'}")
    if test_result['success']:
        print(f"Test Path Length: {test_result['path_length']}")
    print()


def demo_algorithm_comparison():
    """Demonstrate algorithm comparison."""
    print("=" * 60)
    print("🌀 MAZEMIND - Algorithm Comparison Demo")
    print("=" * 60)
    
    simulator = MazeSimulator()
    
    print("Running algorithm comparison...")
    print("This may take a moment...")
    print()
    
    results = simulator.compare_algorithms(
        maze_sizes=[11, 15], 
        num_trials=5
    )
    
    print("Comparison Results:")
    print("-" * 50)
    
    algorithms = ['bfs', 'dfs', 'a_star', 'dijkstra']
    
    for size in results:
        print(f"\nMaze Size: {size}x{size}")
        print(f"{'Algorithm':<12} {'Success Rate':<12} {'Avg Path Length':<15} {'Avg Time (ms)':<15}")
        print("-" * 50)
        
        for algorithm in algorithms:
            trials = results[size][algorithm]
            success_rate = sum(1 for r in trials if r['success']) / len(trials)
            
            successful_trials = [r for r in trials if r['success']]
            if successful_trials:
                avg_length = np.mean([r['path_length'] for r in successful_trials])
                avg_time = np.mean([r['execution_time'] for r in trials]) * 1000
            else:
                avg_length = 0
                avg_time = np.mean([r['execution_time'] for r in trials]) * 1000
            
            print(f"{algorithm.upper():<12} {success_rate:<12.3f} {avg_length:<15.1f} {avg_time:<15.2f}")
    
    print()


def main():
    """Run all demos."""
    print("🌀 Welcome to MazeMind Demo!")
    print("This demo showcases the key features of the MazeMind system.")
    print()
    
    try:
        # Run demos
        demo_classical_algorithms()
        demo_rl_training()
        demo_algorithm_comparison()
        
        print("=" * 60)
        print("🎉 Demo completed successfully!")
        print("=" * 60)
        print()
        print("Key Features Demonstrated:")
        print("✓ Multiple maze generation algorithms")
        print("✓ Classical pathfinding algorithms (BFS, DFS, A*, Dijkstra)")
        print("✓ Q-Learning reinforcement learning agent")
        print("✓ Performance comparison and analysis")
        print("✓ Real-time visualization capabilities")
        print()
        print("For more features, run: python simulate.py")
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\nError during demo: {e}")


if __name__ == "__main__":
    main()
