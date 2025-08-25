"""
Tests for MazeMind - Pathfinding Module
"""

import pytest
import numpy as np
from pathfinding import PathFinder


class TestPathFinder:
    """Test cases for PathFinder class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.pathfinder = PathFinder()
        
        # Create simple test maze
        self.simple_maze = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 0, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0]
        ])
        
        # Create larger test maze
        self.large_maze = np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0]
        ])
    
    def test_init(self):
        """Test PathFinder initialization."""
        assert len(self.pathfinder.directions) == 4
        assert (-1, 0) in self.pathfinder.directions  # up
        assert (1, 0) in self.pathfinder.directions   # down
        assert (0, -1) in self.pathfinder.directions  # left
        assert (0, 1) in self.pathfinder.directions   # right
    
    def test_bfs_simple_maze(self):
        """Test BFS on simple maze."""
        start = (1, 1)
        goal = (3, 3)
        
        path = self.pathfinder.bfs(self.simple_maze, start, goal)
        
        assert path is not None
        assert path[0] == start
        assert path[-1] == goal
        assert len(path) > 0
        
        # Check path validity
        for pos in path:
            x, y = pos
            assert self.simple_maze[y, x] == 1
    
    def test_dfs_simple_maze(self):
        """Test DFS on simple maze."""
        start = (1, 1)
        goal = (3, 3)
        
        path = self.pathfinder.dfs(self.simple_maze, start, goal)
        
        assert path is not None
        assert path[0] == start
        assert path[-1] == goal
        assert len(path) > 0
        
        # Check path validity
        for pos in path:
            x, y = pos
            assert self.simple_maze[y, x] == 1
    
    def test_a_star_simple_maze(self):
        """Test A* on simple maze."""
        start = (1, 1)
        goal = (3, 3)
        
        path = self.pathfinder.a_star(self.simple_maze, start, goal)
        
        assert path is not None
        assert path[0] == start
        assert path[-1] == goal
        assert len(path) > 0
        
        # Check path validity
        for pos in path:
            x, y = pos
            assert self.simple_maze[y, x] == 1
    
    def test_dijkstra_simple_maze(self):
        """Test Dijkstra on simple maze."""
        start = (1, 1)
        goal = (3, 3)
        
        path = self.pathfinder.dijkstra(self.simple_maze, start, goal)
        
        assert path is not None
        assert path[0] == start
        assert path[-1] == goal
        assert len(path) > 0
        
        # Check path validity
        for pos in path:
            x, y = pos
            assert self.simple_maze[y, x] == 1
    
    def test_no_path(self):
        """Test when no path exists."""
        # Create maze with no path
        blocked_maze = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0]
        ])
        
        start = (1, 1)
        goal = (3, 3)
        
        # All algorithms should return None
        assert self.pathfinder.bfs(blocked_maze, start, goal) is None
        assert self.pathfinder.dfs(blocked_maze, start, goal) is None
        assert self.pathfinder.a_star(blocked_maze, start, goal) is None
        assert self.pathfinder.dijkstra(blocked_maze, start, goal) is None
    
    def test_invalid_start_goal(self):
        """Test with invalid start/goal positions."""
        start = (0, 0)  # Wall position
        goal = (1, 1)
        
        # Should return None when start is a wall
        assert self.pathfinder.bfs(self.simple_maze, start, goal) is None
        assert self.pathfinder.dfs(self.simple_maze, start, goal) is None
        assert self.pathfinder.a_star(self.simple_maze, start, goal) is None
        assert self.pathfinder.dijkstra(self.simple_maze, start, goal) is None
        
        start = (1, 1)
        goal = (0, 0)  # Wall position
        
        # Should return None when goal is a wall
        assert self.pathfinder.bfs(self.simple_maze, start, goal) is None
        assert self.pathfinder.dfs(self.simple_maze, start, goal) is None
        assert self.pathfinder.a_star(self.simple_maze, start, goal) is None
        assert self.pathfinder.dijkstra(self.simple_maze, start, goal) is None
    
    def test_solve_maze(self):
        """Test solve_maze method."""
        start = (1, 1)
        goal = (3, 3)
        
        algorithms = ['bfs', 'dfs', 'a_star', 'dijkstra']
        
        for algorithm in algorithms:
            result = self.pathfinder.solve_maze(self.simple_maze, start, goal, algorithm)
            
            assert 'algorithm' in result
            assert 'success' in result
            assert 'path' in result
            assert 'path_length' in result
            assert 'execution_time' in result
            assert 'nodes_explored' in result
            
            assert result['algorithm'] == algorithm
            assert result['success'] is True
            assert result['path'] is not None
            assert result['path_length'] > 0
            assert result['execution_time'] >= 0
            assert result['nodes_explored'] > 0
    
    def test_solve_maze_invalid_algorithm(self):
        """Test solve_maze with invalid algorithm."""
        start = (1, 1)
        goal = (3, 3)
        
        with pytest.raises(ValueError):
            self.pathfinder.solve_maze(self.simple_maze, start, goal, "invalid_algorithm")
    
    def test_optimality_comparison(self):
        """Test that A* and Dijkstra find optimal paths."""
        start = (1, 1)
        goal = (5, 5)
        
        # Test on larger maze
        a_star_path = self.pathfinder.a_star(self.large_maze, start, goal)
        dijkstra_path = self.pathfinder.dijkstra(self.large_maze, start, goal)
        
        assert a_star_path is not None
        assert dijkstra_path is not None
        
        # Both should find paths of the same length (optimal)
        assert len(a_star_path) == len(dijkstra_path)
    
    def test_bfs_optimality(self):
        """Test that BFS finds optimal path."""
        start = (1, 1)
        goal = (5, 5)
        
        bfs_path = self.pathfinder.bfs(self.large_maze, start, goal)
        a_star_path = self.pathfinder.a_star(self.large_maze, start, goal)
        
        assert bfs_path is not None
        assert a_star_path is not None
        
        # BFS should find optimal path
        assert len(bfs_path) == len(a_star_path)
    
    def test_dfs_not_necessarily_optimal(self):
        """Test that DFS may not find optimal path."""
        start = (1, 1)
        goal = (5, 5)
        
        dfs_path = self.pathfinder.dfs(self.large_maze, start, goal)
        a_star_path = self.pathfinder.a_star(self.large_maze, start, goal)
        
        assert dfs_path is not None
        assert a_star_path is not None
        
        # DFS path might be longer (not optimal)
        assert len(dfs_path) >= len(a_star_path)
    
    def test_node_exploration_counting(self):
        """Test node exploration counting."""
        start = (1, 1)
        goal = (3, 3)
        
        algorithms = ['bfs', 'dfs', 'a_star', 'dijkstra']
        
        for algorithm in algorithms:
            nodes_explored = self.pathfinder._count_explored_nodes(
                self.simple_maze, start, goal, algorithm
            )
            
            assert nodes_explored > 0
            assert isinstance(nodes_explored, int)
    
    def test_large_maze_performance(self):
        """Test performance on larger maze."""
        start = (1, 1)
        goal = (5, 5)
        
        algorithms = ['bfs', 'dfs', 'a_star', 'dijkstra']
        
        for algorithm in algorithms:
            result = self.pathfinder.solve_maze(self.large_maze, start, goal, algorithm)
            
            assert result['success'] is True
            assert result['execution_time'] < 1.0  # Should complete quickly
            assert result['nodes_explored'] > 0
    
    def test_path_continuity(self):
        """Test that paths are continuous (adjacent positions)."""
        start = (1, 1)
        goal = (3, 3)
        
        algorithms = ['bfs', 'dfs', 'a_star', 'dijkstra']
        
        for algorithm in algorithms:
            path = getattr(self.pathfinder, algorithm)(self.simple_maze, start, goal)
            
            assert path is not None
            
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
