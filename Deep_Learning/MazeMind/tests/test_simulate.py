"""
Tests for MazeMind - Simulation Module
"""

import pytest
import numpy as np
import tempfile
import os
from simulate import MazeSimulator
from maze_generator import MazeGenerator
from pathfinding import PathFinder
from rl_agent import QLearningAgent


class TestMazeSimulator:
    """Test cases for MazeSimulator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.simulator = MazeSimulator()
        
        # Create test maze
        self.test_maze = np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0]
        ])
        
        self.start = (1, 1)
        self.goal = (5, 5)
    
    def test_init(self):
        """Test MazeSimulator initialization."""
        assert self.simulator.figsize == (12, 8)
        assert isinstance(self.simulator.pathfinder, PathFinder)
        assert isinstance(self.simulator.generator, MazeGenerator)
    
    def test_generate_and_solve(self):
        """Test generate_and_solve method."""
        result = self.simulator.generate_and_solve(
            width=11, height=11, 
            algorithm="a_star", 
            generation_method="dfs"
        )
        
        assert 'algorithm' in result
        assert 'success' in result
        assert 'path' in result
        assert 'path_length' in result
        assert 'execution_time' in result
        assert 'nodes_explored' in result
        assert 'maze' in result
        assert 'start' in result
        assert 'goal' in result
        assert 'generation_method' in result
        
        assert result['algorithm'] == 'a_star'
        assert result['generation_method'] == 'dfs'
        assert result['success'] is True
        assert result['path'] is not None
        assert result['path_length'] > 0
        assert result['execution_time'] >= 0
        assert result['nodes_explored'] > 0
        
        # Check maze properties
        maze = result['maze']
        assert maze.shape == (11, 11)
        assert maze.dtype == np.int64
    
    def test_generate_and_solve_invalid_method(self):
        """Test generate_and_solve with invalid generation method."""
        with pytest.raises(ValueError):
            self.simulator.generate_and_solve(
                width=11, height=11,
                algorithm="a_star",
                generation_method="invalid_method"
            )
    
    def test_generate_and_solve_all_algorithms(self):
        """Test generate_and_solve with all algorithms."""
        algorithms = ["bfs", "dfs", "a_star", "dijkstra"]
        generation_methods = ["dfs", "prims", "recursive_division"]
        
        for algorithm in algorithms:
            for method in generation_methods:
                result = self.simulator.generate_and_solve(
                    width=11, height=11,
                    algorithm=algorithm,
                    generation_method=method
                )
                
                assert result['algorithm'] == algorithm
                assert result['generation_method'] == method
                assert result['success'] is True
                assert result['path'] is not None
    
    def test_train_episode(self):
        """Test train_episode method."""
        agent = QLearningAgent()
        
        success, steps = self.simulator.train_episode(
            agent, self.test_maze, self.start, self.goal
        )
        
        assert isinstance(success, bool)
        assert isinstance(steps, int)
        assert steps > 0
    
    def test_train_agent(self):
        """Test train_agent method."""
        agent = QLearningAgent()
        
        results = self.simulator.train_agent(agent, episodes=5, maze_size=11)
        
        assert 'episodes' in results
        assert 'final_success_rate' in results
        assert 'final_avg_steps' in results
        assert 'training_history' in results
        assert 'agent_stats' in results
        
        assert results['episodes'] == 5
        assert 0 <= results['final_success_rate'] <= 1
        assert results['final_avg_steps'] > 0
        assert len(results['training_history']) == 5
        
        # Check training history
        for episode in results['training_history']:
            assert 'episode' in episode
            assert 'success' in episode
            assert 'steps' in episode
    
    def test_compare_algorithms(self):
        """Test compare_algorithms method."""
        results = self.simulator.compare_algorithms(
            maze_sizes=[11, 15], num_trials=3
        )
        
        assert 11 in results
        assert 15 in results
        
        algorithms = ['bfs', 'dfs', 'a_star', 'dijkstra']
        
        for size in results:
            for algorithm in algorithms:
                assert algorithm in results[size]
                assert len(results[size][algorithm]) == 3
                
                # Check each trial result
                for trial in results[size][algorithm]:
                    assert 'algorithm' in trial
                    assert 'success' in trial
                    assert 'path' in trial
                    assert 'path_length' in trial
                    assert 'execution_time' in trial
                    assert 'nodes_explored' in trial
    
    def test_visualize_path(self):
        """Test visualize_path method."""
        # Create a simple path
        path = [(1, 1), (2, 1), (3, 1), (3, 2), (3, 3), (4, 3), (5, 3), (5, 4), (5, 5)]
        
        # Test without saving
        self.simulator.visualize_path(
            self.test_maze, path, self.start, self.goal, "Test Visualization"
        )
        
        # Test with saving
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            filename = tmp_file.name
        
        try:
            self.simulator.visualize_path(
                self.test_maze, path, self.start, self.goal, 
                "Test Visualization", save_path=filename
            )
            
            # Check that file was created
            assert os.path.exists(filename)
            assert os.path.getsize(filename) > 0
        
        finally:
            # Clean up
            if os.path.exists(filename):
                os.unlink(filename)
    
    def test_visualize_path_empty_path(self):
        """Test visualize_path with empty path."""
        # Should work without errors
        self.simulator.visualize_path(
            self.test_maze, [], self.start, self.goal, "Empty Path Test"
        )
    
    def test_animate_solution(self):
        """Test animate_solution method."""
        # Create a simple path
        path = [(1, 1), (2, 1), (3, 1), (3, 2), (3, 3), (4, 3), (5, 3), (5, 4), (5, 5)]
        
        # Test without saving
        anim = self.simulator.animate_solution(
            self.test_maze, path, self.start, self.goal
        )
        
        assert anim is not None
        
        # Test with saving
        with tempfile.NamedTemporaryFile(delete=False, suffix='.gif') as tmp_file:
            filename = tmp_file.name
        
        try:
            anim = self.simulator.animate_solution(
                self.test_maze, path, self.start, self.goal, save_path=filename
            )
            
            # Check that file was created
            assert os.path.exists(filename)
            assert os.path.getsize(filename) > 0
        
        finally:
            # Clean up
            if os.path.exists(filename):
                os.unlink(filename)
    
    def test_plot_comparison(self):
        """Test plot_comparison method."""
        # Create mock comparison results
        results = {
            11: {
                'bfs': [
                    {'success': True, 'path_length': 10, 'execution_time': 0.001, 'nodes_explored': 20},
                    {'success': True, 'path_length': 12, 'execution_time': 0.002, 'nodes_explored': 25}
                ],
                'dfs': [
                    {'success': True, 'path_length': 15, 'execution_time': 0.001, 'nodes_explored': 15},
                    {'success': True, 'path_length': 18, 'execution_time': 0.002, 'nodes_explored': 18}
                ],
                'a_star': [
                    {'success': True, 'path_length': 10, 'execution_time': 0.001, 'nodes_explored': 12},
                    {'success': True, 'path_length': 12, 'execution_time': 0.002, 'nodes_explored': 14}
                ],
                'dijkstra': [
                    {'success': True, 'path_length': 10, 'execution_time': 0.002, 'nodes_explored': 20},
                    {'success': True, 'path_length': 12, 'execution_time': 0.003, 'nodes_explored': 25}
                ]
            },
            15: {
                'bfs': [
                    {'success': True, 'path_length': 15, 'execution_time': 0.002, 'nodes_explored': 30},
                    {'success': True, 'path_length': 18, 'execution_time': 0.003, 'nodes_explored': 35}
                ],
                'dfs': [
                    {'success': True, 'path_length': 22, 'execution_time': 0.002, 'nodes_explored': 25},
                    {'success': True, 'path_length': 25, 'execution_time': 0.003, 'nodes_explored': 28}
                ],
                'a_star': [
                    {'success': True, 'path_length': 15, 'execution_time': 0.002, 'nodes_explored': 20},
                    {'success': True, 'path_length': 18, 'execution_time': 0.003, 'nodes_explored': 22}
                ],
                'dijkstra': [
                    {'success': True, 'path_length': 15, 'execution_time': 0.003, 'nodes_explored': 30},
                    {'success': True, 'path_length': 18, 'execution_time': 0.004, 'nodes_explored': 35}
                ]
            }
        }
        
        # Should not raise any errors
        self.simulator.plot_comparison(results)
    
    def test_plot_training_progress(self):
        """Test plot_training_progress method."""
        # Create mock training results
        training_results = {
            'episodes': 10,
            'final_success_rate': 0.8,
            'final_avg_steps': 25.5,
            'training_history': [
                {'episode': i, 'success': i % 2 == 0, 'steps': 20 + i}
                for i in range(10)
            ],
            'agent_stats': {
                'total_episodes': 10,
                'average_reward': -5.2,
                'average_steps': 25.5,
                'current_epsilon': 0.05,
                'q_table_size': 150
            }
        }
        
        # Should not raise any errors
        self.simulator.plot_training_progress(training_results)
    
    def test_different_maze_sizes(self):
        """Test with different maze sizes."""
        sizes = [11, 15, 21]
        
        for size in sizes:
            result = self.simulator.generate_and_solve(
                width=size, height=size,
                algorithm="a_star",
                generation_method="dfs"
            )
            
            assert result['maze'].shape == (size, size)
            assert result['success'] is True
    
    def test_all_generation_methods(self):
        """Test all generation methods."""
        methods = ["dfs", "prims", "recursive_division"]
        
        for method in methods:
            result = self.simulator.generate_and_solve(
                width=11, height=11,
                algorithm="a_star",
                generation_method=method
            )
            
            assert result['generation_method'] == method
            assert result['success'] is True
    
    def test_performance_metrics(self):
        """Test that performance metrics are reasonable."""
        result = self.simulator.generate_and_solve(
            width=15, height=15,
            algorithm="a_star",
            generation_method="dfs"
        )
        
        # Execution time should be reasonable
        assert result['execution_time'] < 1.0  # Should complete quickly
        
        # Path length should be reasonable for 15x15 maze
        assert result['path_length'] > 0
        assert result['path_length'] < 100  # Should not be excessively long
        
        # Nodes explored should be reasonable
        assert result['nodes_explored'] > 0
        assert result['nodes_explored'] < 1000  # Should not explore too many nodes
    
    def test_path_validity(self):
        """Test that generated paths are valid."""
        result = self.simulator.generate_and_solve(
            width=11, height=11,
            algorithm="a_star",
            generation_method="dfs"
        )
        
        path = result['path']
        maze = result['maze']
        
        # Check that path starts and ends correctly
        assert path[0] == result['start']
        assert path[-1] == result['goal']
        
        # Check that all path positions are valid
        for pos in path:
            x, y = pos
            assert 0 <= x < maze.shape[1]
            assert 0 <= y < maze.shape[0]
            assert maze[y, x] == 1  # Should be a path, not a wall
        
        # Check that consecutive positions are adjacent
        for i in range(len(path) - 1):
            pos1 = path[i]
            pos2 = path[i + 1]
            
            dx = abs(pos1[0] - pos2[0])
            dy = abs(pos1[1] - pos2[1])
            
            # Should be adjacent (Manhattan distance = 1)
            assert dx + dy == 1


if __name__ == "__main__":
    pytest.main([__file__])
