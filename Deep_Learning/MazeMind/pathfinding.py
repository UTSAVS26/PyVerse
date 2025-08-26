"""
MazeMind - Pathfinding Module

This module provides classical pathfinding algorithms:
- Breadth-First Search (BFS)
- Depth-First Search (DFS)
- A* Search Algorithm
"""

import numpy as np
import heapq
from typing import Tuple, List, Optional, Dict, Set
from collections import deque
import time


class PathFinder:
    """Implements various pathfinding algorithms for maze solving."""
    
    def __init__(self):
        """Initialize pathfinder."""
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
    
    def bfs(self, maze: np.ndarray, start: Tuple[int, int], 
            goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Breadth-First Search algorithm.
        
        Args:
            maze: 2D numpy array where 0 = wall, 1 = path
            start: Starting position (x, y)
            goal: Goal position (x, y)
            
        Returns:
            List of positions from start to goal, or None if no path exists
        """
        if maze[start[1], start[0]] == 0 or maze[goal[1], goal[0]] == 0:
            return None
        
        queue = deque([(start, [start])])
        visited = {start}
        
        while queue:
            current, path = queue.popleft()
            
            if current == goal:
                return path
            
            for dx, dy in self.directions:
                next_x, next_y = current[0] + dx, current[1] + dy
                next_pos = (next_x, next_y)
                
                if (0 <= next_x < maze.shape[1] and 
                    0 <= next_y < maze.shape[0] and
                    maze[next_y, next_x] == 1 and
                    next_pos not in visited):
                    
                    visited.add(next_pos)
                    new_path = path + [next_pos]
                    queue.append((next_pos, new_path))
        
        return None
    
    def dfs(self, maze: np.ndarray, start: Tuple[int, int], 
            goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Depth-First Search algorithm.
        
        Args:
            maze: 2D numpy array where 0 = wall, 1 = path
            start: Starting position (x, y)
            goal: Goal position (x, y)
            
        Returns:
            List of positions from start to goal, or None if no path exists
        """
        if maze[start[1], start[0]] == 0 or maze[goal[1], goal[0]] == 0:
            return None
        
        stack = [(start, [start])]
        visited = {start}
        
        while stack:
            current, path = stack.pop()
            
            if current == goal:
                return path
            
            for dx, dy in self.directions:
                next_x, next_y = current[0] + dx, current[1] + dy
                next_pos = (next_x, next_y)
                
                if (0 <= next_x < maze.shape[1] and 
                    0 <= next_y < maze.shape[0] and
                    maze[next_y, next_x] == 1 and
                    next_pos not in visited):
                    
                    visited.add(next_pos)
                    new_path = path + [next_pos]
                    stack.append((next_pos, new_path))
        
        return None
    
    def a_star(self, maze: np.ndarray, start: Tuple[int, int], 
               goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        A* Search algorithm with Manhattan distance heuristic.
        
        Args:
            maze: 2D numpy array where 0 = wall, 1 = path
            start: Starting position (x, y)
            goal: Goal position (x, y)
            
        Returns:
            List of positions from start to goal, or None if no path exists
        """
        if maze[start[1], start[0]] == 0 or maze[goal[1], goal[0]] == 0:
            return None
        
        def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
            """Calculate Manhattan distance between two positions."""
            return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
        
        # Priority queue: (f_score, current_pos, g_score, path)
        open_set = [(manhattan_distance(start, goal), start, 0, [start])]
        closed_set = set()
        g_scores = {start: 0}
        
        while open_set:
            f_score, current, g_score, path = heapq.heappop(open_set)
            
            if current in closed_set:
                continue
            
            if current == goal:
                return path
            
            closed_set.add(current)
            
            for dx, dy in self.directions:
                next_x, next_y = current[0] + dx, current[1] + dy
                next_pos = (next_x, next_y)
                
                if (0 <= next_x < maze.shape[1] and 
                    0 <= next_y < maze.shape[0] and
                    maze[next_y, next_x] == 1 and
                    next_pos not in closed_set):
                    
                    new_g_score = g_score + 1
                    
                    if next_pos not in g_scores or new_g_score < g_scores[next_pos]:
                        g_scores[next_pos] = new_g_score
                        new_f_score = new_g_score + manhattan_distance(next_pos, goal)
                        new_path = path + [next_pos]
                        heapq.heappush(open_set, (new_f_score, next_pos, new_g_score, new_path))
        
        return None
    
    def dijkstra(self, maze: np.ndarray, start: Tuple[int, int], 
                 goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Dijkstra's algorithm for finding shortest path.
        
        Args:
            maze: 2D numpy array where 0 = wall, 1 = path
            start: Starting position (x, y)
            goal: Goal position (x, y)
            
        Returns:
            List of positions from start to goal, or None if no path exists
        """
        if maze[start[1], start[0]] == 0 or maze[goal[1], goal[0]] == 0:
            return None
        
        # Priority queue: (distance, current_pos, path)
        open_set = [(0, start, [start])]
        closed_set = set()
        distances = {start: 0}
        
        while open_set:
            distance, current, path = heapq.heappop(open_set)
            
            if current in closed_set:
                continue
            
            if current == goal:
                return path
            
            closed_set.add(current)
            
            for dx, dy in self.directions:
                next_x, next_y = current[0] + dx, current[1] + dy
                next_pos = (next_x, next_y)
                
                if (0 <= next_x < maze.shape[1] and 
                    0 <= next_y < maze.shape[0] and
                    maze[next_y, next_x] == 1 and
                    next_pos not in closed_set):
                    
                    new_distance = distance + 1
                    
                    if next_pos not in distances or new_distance < distances[next_pos]:
                        distances[next_pos] = new_distance
                        new_path = path + [next_pos]
                        heapq.heappush(open_set, (new_distance, next_pos, new_path))
        
        return None
    
    def solve_maze(self, maze: np.ndarray, start: Tuple[int, int], 
                   goal: Tuple[int, int], algorithm: str = "a_star") -> Dict:
        """
        Solve maze using specified algorithm and return performance metrics.
        
        Args:
            maze: 2D numpy array where 0 = wall, 1 = path
            start: Starting position (x, y)
            goal: Goal position (x, y)
            algorithm: Algorithm to use ("bfs", "dfs", "a_star", "dijkstra")
            
        Returns:
            Dictionary with path, performance metrics, and success status
        """
        algorithms = {
            "bfs": self.bfs,
            "dfs": self.dfs,
            "a_star": self.a_star,
            "dijkstra": self.dijkstra
        }
        
        if algorithm not in algorithms:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        start_time = time.time()
        path = algorithms[algorithm](maze, start, goal)
        end_time = time.time()
        
        result = {
            "algorithm": algorithm,
            "success": path is not None,
            "path": path,
            "path_length": len(path) if path else 0,
            "execution_time": end_time - start_time,
            "nodes_explored": self._count_explored_nodes(maze, start, goal, algorithm)
        }
        
        return result
    
    def _count_explored_nodes(self, maze: np.ndarray, start: Tuple[int, int], 
                             goal: Tuple[int, int], algorithm: str) -> int:
        """
        Count number of nodes explored by the algorithm.
        
        Args:
            maze: 2D numpy array where 0 = wall, 1 = path
            start: Starting position (x, y)
            goal: Goal position (x, y)
            algorithm: Algorithm to use
            
        Returns:
            Number of nodes explored
        """
        if algorithm == "bfs":
            return self._count_bfs_nodes(maze, start, goal)
        elif algorithm == "dfs":
            return self._count_dfs_nodes(maze, start, goal)
        elif algorithm == "a_star":
            return self._count_a_star_nodes(maze, start, goal)
        elif algorithm == "dijkstra":
            return self._count_dijkstra_nodes(maze, start, goal)
        else:
            return 0
    
    def _count_bfs_nodes(self, maze: np.ndarray, start: Tuple[int, int], 
                        goal: Tuple[int, int]) -> int:
        """Count nodes explored by BFS."""
        if maze[start[1], start[0]] == 0 or maze[goal[1], goal[0]] == 0:
            return 0
        
        queue = deque([start])
        visited = {start}
        count = 1
        
        while queue:
            current = queue.popleft()
            
            if current == goal:
                return count
            
            for dx, dy in self.directions:
                next_x, next_y = current[0] + dx, current[1] + dy
                next_pos = (next_x, next_y)
                
                if (0 <= next_x < maze.shape[1] and 
                    0 <= next_y < maze.shape[0] and
                    maze[next_y, next_x] == 1 and
                    next_pos not in visited):
                    
                    visited.add(next_pos)
                    queue.append(next_pos)
                    count += 1
        
        return count
    
    def _count_dfs_nodes(self, maze: np.ndarray, start: Tuple[int, int], 
                        goal: Tuple[int, int]) -> int:
        """Count nodes explored by DFS."""
        if maze[start[1], start[0]] == 0 or maze[goal[1], goal[0]] == 0:
            return 0
        
        stack = [start]
        visited = {start}
        count = 1
        
        while stack:
            current = stack.pop()
            
            if current == goal:
                return count
            
            for dx, dy in self.directions:
                next_x, next_y = current[0] + dx, current[1] + dy
                next_pos = (next_x, next_y)
                
                if (0 <= next_x < maze.shape[1] and 
                    0 <= next_y < maze.shape[0] and
                    maze[next_y, next_x] == 1 and
                    next_pos not in visited):
                    
                    visited.add(next_pos)
                    stack.append(next_pos)
                    count += 1
        
        return count
    
    def _count_a_star_nodes(self, maze: np.ndarray, start: Tuple[int, int], 
                           goal: Tuple[int, int]) -> int:
        """Count nodes explored by A*."""
        if maze[start[1], start[0]] == 0 or maze[goal[1], goal[0]] == 0:
            return 0
        
        def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
            return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
        
        open_set = [(manhattan_distance(start, goal), start)]
        closed_set = set()
        g_scores = {start: 0}
        count = 0
        
        while open_set:
            f_score, current = heapq.heappop(open_set)
            
            if current in closed_set:
                continue
            
            count += 1
            
            if current == goal:
                return count
            
            closed_set.add(current)
            
            for dx, dy in self.directions:
                next_x, next_y = current[0] + dx, current[1] + dy
                next_pos = (next_x, next_y)
                
                if (0 <= next_x < maze.shape[1] and 
                    0 <= next_y < maze.shape[0] and
                    maze[next_y, next_x] == 1 and
                    next_pos not in closed_set):
                    
                    new_g_score = g_scores[current] + 1
                    
                    if next_pos not in g_scores or new_g_score < g_scores[next_pos]:
                        g_scores[next_pos] = new_g_score
                        new_f_score = new_g_score + manhattan_distance(next_pos, goal)
                        heapq.heappush(open_set, (new_f_score, next_pos))
        
        return count
    
    def _count_dijkstra_nodes(self, maze: np.ndarray, start: Tuple[int, int], 
                             goal: Tuple[int, int]) -> int:
        """Count nodes explored by Dijkstra."""
        if maze[start[1], start[0]] == 0 or maze[goal[1], goal[0]] == 0:
            return 0
        
        open_set = [(0, start)]
        closed_set = set()
        distances = {start: 0}
        count = 0
        
        while open_set:
            distance, current = heapq.heappop(open_set)
            
            if current in closed_set:
                continue
            
            count += 1
            
            if current == goal:
                return count
            
            closed_set.add(current)
            
            for dx, dy in self.directions:
                next_x, next_y = current[0] + dx, current[1] + dy
                next_pos = (next_x, next_y)
                
                if (0 <= next_x < maze.shape[1] and 
                    0 <= next_y < maze.shape[0] and
                    maze[next_y, next_x] == 1 and
                    next_pos not in closed_set):
                    
                    new_distance = distance + 1
                    
                    if next_pos not in distances or new_distance < distances[next_pos]:
                        distances[next_pos] = new_distance
                        heapq.heappush(open_set, (new_distance, next_pos))
        
        return count
