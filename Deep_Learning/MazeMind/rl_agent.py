"""
MazeMind - Reinforcement Learning Agent Module

This module provides Q-learning agent for adaptive maze exploration.
"""

import numpy as np
import random
from typing import Tuple, List, Dict, Optional
from collections import defaultdict
import time


class QLearningAgent:
    """Q-learning agent for maze navigation."""
    
    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.9, 
                 epsilon: float = 0.1, epsilon_decay: float = 0.995, 
                 min_epsilon: float = 0.01):
        """
        Initialize Q-learning agent.
        
        Args:
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            epsilon: Initial exploration rate
            epsilon_decay: Epsilon decay rate
            min_epsilon: Minimum epsilon value
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        
        # Q-table: state -> action -> value
        self.q_table = defaultdict(lambda: defaultdict(float))
        
        # Actions: up, right, down, left
        self.actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        self.action_names = ['up', 'right', 'down', 'left']
        
        # Training statistics
        self.episode_rewards = []
        self.episode_steps = []
        self.success_rate = []
        
    def get_state(self, position: Tuple[int, int], maze: np.ndarray) -> str:
        """
        Convert position and local maze view to state representation.
        
        Args:
            position: Current position (x, y)
            maze: Maze grid
            
        Returns:
            State string representation
        """
        x, y = position
        height, width = maze.shape
        
        # Get local view (3x3 around current position)
        local_view = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    local_view.append(str(maze[ny, nx]))
                else:
                    local_view.append('0')  # Wall outside bounds
        
        # Add relative position to goal (simplified)
        goal_x, goal_y = width - 2, height - 2
        relative_x = 'left' if x < goal_x else 'right' if x > goal_x else 'center'
        relative_y = 'up' if y < goal_y else 'down' if y > goal_y else 'center'
        
        state = f"{''.join(local_view)}_{relative_x}_{relative_y}"
        return state
    
    def choose_action(self, state: str, training: bool = True) -> int:
        """
        Choose action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether in training mode
            
        Returns:
            Action index
        """
        if training and random.random() < self.epsilon:
            # Exploration: random action
            return random.randint(0, len(self.actions) - 1)
        else:
            # Exploitation: best action
            q_values = [self.q_table[state][i] for i in range(len(self.actions))]
            max_q = max(q_values)
            best_actions = [i for i, q in enumerate(q_values) if q == max_q]
            return random.choice(best_actions)
    
    def get_reward(self, position: Tuple[int, int], maze: np.ndarray, 
                   goal: Tuple[int, int], step_count: int, max_steps: int) -> float:
        """
        Calculate reward for current state.
        
        Args:
            position: Current position
            maze: Maze grid
            goal: Goal position
            step_count: Current step count
            max_steps: Maximum steps allowed
            
        Returns:
            Reward value
        """
        x, y = position
        
        # Check if out of bounds or hit wall
        if (x < 0 or x >= maze.shape[1] or 
            y < 0 or y >= maze.shape[0] or 
            maze[y, x] == 0):
            return -100  # Penalty for invalid move
        
        # Check if reached goal
        if position == goal:
            return 100  # Large reward for reaching goal
        
        # Check if exceeded max steps
        if step_count >= max_steps:
            return -50  # Penalty for timeout
        
        # Distance-based reward
        current_distance = abs(x - goal[0]) + abs(y - goal[1])
        return -current_distance * 0.1  # Small penalty for distance
    
    def update_q_value(self, state: str, action: int, reward: float, 
                      next_state: str, next_action: int = None):
        """
        Update Q-value using Q-learning update rule.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            next_action: Next action (for SARSA)
        """
        current_q = self.q_table[state][action]
        
        if next_action is not None:
            # SARSA update
            next_q = self.q_table[next_state][next_action]
        else:
            # Q-learning update (off-policy)
            next_q = max(self.q_table[next_state].values()) if self.q_table[next_state] else 0
        
        # Q-learning update rule
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * next_q - current_q)
        self.q_table[state][action] = new_q
    
    def train_episode(self, maze: np.ndarray, start: Tuple[int, int], 
                     goal: Tuple[int, int], max_steps: int = 1000) -> Dict:
        """
        Train agent for one episode.
        
        Args:
            maze: Maze grid
            start: Starting position
            goal: Goal position
            max_steps: Maximum steps per episode
            
        Returns:
            Episode results dictionary
        """
        position = start
        total_reward = 0
        steps = 0
        path = [start]
        
        while steps < max_steps:
            # Get current state
            state = self.get_state(position, maze)
            
            # Choose action
            action = self.choose_action(state, training=True)
            dx, dy = self.actions[action]
            
            # Take action
            next_position = (position[0] + dx, position[1] + dy)
            
            # Get reward
            reward = self.get_reward(next_position, maze, goal, steps, max_steps)
            
            # Get next state
            next_state = self.get_state(next_position, maze)
            
            # Update Q-value
            self.update_q_value(state, action, reward, next_state)
            
            # Update position and statistics
            if reward > -100:  # Valid move
                position = next_position
                path.append(position)
            
            total_reward += reward
            steps += 1
            
            # Check if goal reached
            if position == goal:
                break
        
        # Update epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
        # Record episode statistics
        success = position == goal
        self.episode_rewards.append(total_reward)
        self.episode_steps.append(steps)
        
        return {
            'success': success,
            'total_reward': total_reward,
            'steps': steps,
            'path': path,
            'final_position': position
        }
    
    def solve_maze(self, maze: np.ndarray, start: Tuple[int, int], 
                   goal: Tuple[int, int], max_steps: int = 1000) -> Dict:
        """
        Solve maze using trained Q-values (no exploration).
        
        Args:
            maze: Maze grid
            start: Starting position
            goal: Goal position
            max_steps: Maximum steps allowed
            
        Returns:
            Solution results dictionary
        """
        position = start
        steps = 0
        path = [start]
        visited = {start}
        
        while steps < max_steps:
            # Get current state
            state = self.get_state(position, maze)
            
            # Choose best action (no exploration)
            action = self.choose_action(state, training=False)
            dx, dy = self.actions[action]
            
            # Take action
            next_position = (position[0] + dx, position[1] + dy)
            
            # Check if valid move
            if (0 <= next_position[0] < maze.shape[1] and 
                0 <= next_position[1] < maze.shape[0] and 
                maze[next_position[1], next_position[0]] == 1):
                
                position = next_position
                path.append(position)
                
                # Check if goal reached
                if position == goal:
                    break
                
                # Check for loops
                if position in visited:
                    break
                visited.add(position)
            
            steps += 1
        
        success = position == goal
        return {
            'success': success,
            'steps': steps,
            'path': path,
            'final_position': position
        }
    
    def get_training_stats(self) -> Dict:
        """
        Get training statistics.
        
        Returns:
            Dictionary with training statistics
        """
        if not self.episode_rewards:
            return {}
        
        recent_rewards = self.episode_rewards[-100:] if len(self.episode_rewards) >= 100 else self.episode_rewards
        recent_steps = self.episode_steps[-100:] if len(self.episode_steps) >= 100 else self.episode_steps
        
        return {
            'total_episodes': len(self.episode_rewards),
            'average_reward': np.mean(recent_rewards),
            'average_steps': np.mean(recent_steps),
            'current_epsilon': self.epsilon,
            'q_table_size': len(self.q_table)
        }
    
    def save_q_table(self, filename: str):
        """
        Save Q-table to file.
        
        Args:
            filename: Output filename
        """
        import pickle
        # Convert defaultdict to regular dict for saving
        q_table_dict = {}
        for state, actions in self.q_table.items():
            # Only save states that have actual Q-values
            state_dict = {}
            for action, value in actions.items():
                if value != 0.0:  # Only save non-zero values
                    state_dict[action] = value
            if state_dict:  # Only save states with non-zero values
                q_table_dict[state] = state_dict
        with open(filename, 'wb') as f:
            pickle.dump(q_table_dict, f)
    
    def load_q_table(self, filename: str):
        """
        Load Q-table from file.
        
        Args:
            filename: Input filename
        """
        import pickle
        with open(filename, 'rb') as f:
            q_table_dict = pickle.load(f)
            self.q_table = defaultdict(lambda: defaultdict(float))
            for state, actions in q_table_dict.items():
                for action, value in actions.items():
                    self.q_table[state][action] = value


class MultiAgentSystem:
    """System for managing multiple agents with different strategies."""
    
    def __init__(self):
        """Initialize multi-agent system."""
        self.agents = {}
    
    def add_agent(self, name: str, agent: QLearningAgent):
        """
        Add agent to the system.
        
        Args:
            name: Agent name
            agent: Q-learning agent
        """
        self.agents[name] = agent
    
    def train_all_agents(self, maze: np.ndarray, start: Tuple[int, int], 
                        goal: Tuple[int, int], episodes: int = 100) -> Dict:
        """
        Train all agents on the same maze.
        
        Args:
            maze: Maze grid
            start: Starting position
            goal: Goal position
            episodes: Number of training episodes
            
        Returns:
            Training results for all agents
        """
        results = {}
        
        for name, agent in self.agents.items():
            print(f"Training agent: {name}")
            agent_results = []
            
            for episode in range(episodes):
                result = agent.train_episode(maze, start, goal)
                agent_results.append(result)
                
                if episode % 50 == 0:
                    stats = agent.get_training_stats()
                    print(f"  Episode {episode}: Success={result['success']}, "
                          f"Steps={result['steps']}, Reward={result['total_reward']:.2f}")
            
            results[name] = agent_results
        
        return results
    
    def compare_agents(self, maze: np.ndarray, start: Tuple[int, int], 
                      goal: Tuple[int, int]) -> Dict:
        """
        Compare all agents on the same maze.
        
        Args:
            maze: Maze grid
            start: Starting position
            goal: Goal position
            
        Returns:
            Comparison results
        """
        results = {}
        
        for name, agent in self.agents.items():
            result = agent.solve_maze(maze, start, goal)
            results[name] = result
        
        return results
