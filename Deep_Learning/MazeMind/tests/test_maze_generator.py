"""
Tests for MazeMind - Maze Generator Module
"""

import pytest
import numpy as np
from maze_generator import MazeGenerator


class TestMazeGenerator:
    """Test cases for MazeGenerator class."""
    
    def test_init(self):
        """Test MazeGenerator initialization."""
        # Test with odd dimensions
        generator = MazeGenerator(21, 21)
        assert generator.width == 21
        assert generator.height == 21
        assert generator.complexity == 0.5
        
        # Test with even dimensions (should be converted to odd)
        generator = MazeGenerator(20, 20)
        assert generator.width == 21
        assert generator.height == 21
        
        # Test complexity bounds
        generator = MazeGenerator(21, 21, complexity=1.5)
        assert generator.complexity == 1.0
        
        generator = MazeGenerator(21, 21, complexity=-0.5)
        assert generator.complexity == 0.0
    
    def test_generate_dfs(self):
        """Test DFS maze generation."""
        generator = MazeGenerator(11, 11)
        maze = generator.generate_dfs()
        
        # Check maze dimensions
        assert maze.shape == (11, 11)
        assert maze.dtype == np.int64
        
        # Check that start and end points are accessible
        start, end = generator.get_start_end_points()
        assert maze[start[1], start[0]] == 1
        assert maze[end[1], end[0]] == 1
        
        # Check that maze is valid
        assert generator.validate_maze(maze)
    
    def test_generate_prims(self):
        """Test Prim's algorithm maze generation."""
        generator = MazeGenerator(11, 11)
        maze = generator.generate_prims()
        
        # Check maze dimensions
        assert maze.shape == (11, 11)
        assert maze.dtype == np.int64
        
        # Check that start and end points are accessible
        start, end = generator.get_start_end_points()
        assert maze[start[1], start[0]] == 1
        assert maze[end[1], end[0]] == 1
        
        # Check that maze is valid
        assert generator.validate_maze(maze)
    
    def test_generate_recursive_division(self):
        """Test recursive division maze generation."""
        generator = MazeGenerator(11, 11)
        maze = generator.generate_recursive_division()
        
        # Check maze dimensions
        assert maze.shape == (11, 11)
        assert maze.dtype == np.int64
        
        # Check that start and end points are accessible
        start, end = generator.get_start_end_points()
        assert maze[start[1], start[0]] == 1
        assert maze[end[1], end[0]] == 1
        
        # Check that maze is valid
        assert generator.validate_maze(maze)
    
    def test_get_start_end_points(self):
        """Test start and end point calculation."""
        generator = MazeGenerator(21, 21)
        start, end = generator.get_start_end_points()
        
        assert start == (1, 1)
        assert end == (19, 19)
    
    def test_validate_maze(self):
        """Test maze validation."""
        generator = MazeGenerator(11, 11)
        
        # Test valid maze
        maze = generator.generate_dfs()
        assert generator.validate_maze(maze)
        
        # Test invalid maze (blocked start)
        invalid_maze = maze.copy()
        invalid_maze[1, 1] = 0
        assert not generator.validate_maze(invalid_maze)
        
        # Test invalid maze (blocked end)
        invalid_maze = maze.copy()
        invalid_maze[9, 9] = 0
        assert not generator.validate_maze(invalid_maze)
    
    def test_add_complexity(self):
        """Test complexity addition."""
        generator = MazeGenerator(11, 11, complexity=0.0)
        maze = generator.generate_dfs()
        original_maze = maze.copy()
        
        # Test with zero complexity
        modified_maze = generator.add_complexity(maze)
        assert np.array_equal(original_maze, modified_maze)
        
        # Test with non-zero complexity
        generator.complexity = 0.5
        modified_maze = generator.add_complexity(maze)
        # Should still be valid
        assert generator.validate_maze(modified_maze)
    
    def test_is_connected(self):
        """Test connectivity checking."""
        generator = MazeGenerator(11, 11)
        
        # Test connected maze
        maze = generator.generate_dfs()
        assert generator._is_connected(maze)
        
        # Test disconnected maze
        disconnected_maze = np.zeros((11, 11), dtype=int)
        disconnected_maze[1, 1] = 1  # Start
        disconnected_maze[9, 9] = 1  # End
        # No path between them
        assert not generator._is_connected(disconnected_maze)
    
    def test_different_sizes(self):
        """Test maze generation with different sizes."""
        sizes = [5, 11, 21, 31]
        
        for size in sizes:
            generator = MazeGenerator(size, size)
            maze = generator.generate_dfs()
            
            assert maze.shape == (size, size)
            assert generator.validate_maze(maze)
    
    def test_all_generation_methods(self):
        """Test all generation methods produce valid mazes."""
        generator = MazeGenerator(15, 15)
        
        methods = [
            generator.generate_dfs,
            generator.generate_prims,
            generator.generate_recursive_division
        ]
        
        for method in methods:
            maze = method()
            assert maze.shape == (15, 15)
            assert generator.validate_maze(maze)
            
            # Check that start and end are accessible
            start, end = generator.get_start_end_points()
            assert maze[start[1], start[0]] == 1
            assert maze[end[1], end[0]] == 1


if __name__ == "__main__":
    pytest.main([__file__])
