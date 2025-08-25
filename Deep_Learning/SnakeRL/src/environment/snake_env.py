import pygame
import numpy as np
import random
from typing import Tuple, List, Dict, Any, Optional
from enum import Enum

class Direction(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

class SnakeEnv:
    """
    Snake Game Environment for Reinforcement Learning
    
    Features:
    - Configurable grid size
    - State representation for RL agents
    - Reward system (food: +10, collision: -10, step: -0.1)
    - Pygame visualization
    """
    
    def __init__(self, grid_size: int = 10, cell_size: int = 30, 
                 render_mode: str = "human", fps: int = 10):
        """
        Initialize Snake environment
        
        Args:
            grid_size: Size of the grid (grid_size x grid_size)
            cell_size: Size of each cell in pixels
            render_mode: "human" for pygame display, "rgb_array" for array
            fps: Frames per second for rendering
        """
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.render_mode = render_mode
        self.fps = fps
        
        # Game state
        self.snake = [(grid_size // 2, grid_size // 2)]
        self.direction = Direction.RIGHT
        self.food = self._generate_food()
        self.score = 0
        self.game_over = False
        self.steps = 0
        
        # Pygame setup
        if render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode(
                (grid_size * cell_size, grid_size * cell_size)
            )
            pygame.display.set_caption("Snake Game AI")
            self.clock = pygame.time.Clock()
            
            # Colors
            self.BLACK = (0, 0, 0)
            self.WHITE = (255, 255, 255)
            self.GREEN = (0, 255, 0)
            self.RED = (255, 0, 0)
            self.BLUE = (0, 0, 255)
    
    def reset(self) -> np.ndarray:
        """Reset the environment to initial state"""
        self.snake = [(self.grid_size // 2, self.grid_size // 2)]
        self.direction = Direction.RIGHT
        self.food = self._generate_food()
        self.score = 0
        self.game_over = False
        self.steps = 0
        
        if self.render_mode == "human":
            self.screen.fill(self.BLACK)
            pygame.display.flip()
        
        return self._get_state()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take a step in the environment
        
        Args:
            action: 0=left, 1=straight, 2=right (relative to current direction)
            
        Returns:
            state: Current state representation
            reward: Reward for this step
            done: Whether episode is finished
            info: Additional information
        """
        if self.game_over:
            return self._get_state(), 0, True, {"score": self.score}
        
        # Update direction based on action
        self._update_direction(action)
        
        # Move snake
        new_head = self._get_new_head()
        
        # Check collision
        if self._is_collision(new_head):
            self.game_over = True
            reward = -10.0
        else:
            self.snake.insert(0, new_head)
            
            # Check if food eaten
            if new_head == self.food:
                self.score += 1
                self.food = self._generate_food()
                reward = 10.0
            else:
                self.snake.pop()
                reward = -0.1
            
            self.steps += 1
        
        # Render if needed
        if self.render_mode == "human":
            self._render()
        
        info = {
            "score": self.score,
            "steps": self.steps,
            "snake_length": len(self.snake)
        }
        
        return self._get_state(), reward, self.game_over, info
    
    def _update_direction(self, action: int):
        """Update snake direction based on action"""
        if action == 0:  # Left
            self.direction = Direction((self.direction.value - 1) % 4)
        elif action == 2:  # Right
            self.direction = Direction((self.direction.value + 1) % 4)
        # action == 1 means straight, no change needed
    
    def _get_new_head(self) -> Tuple[int, int]:
        """Get new head position based on current direction"""
        head = self.snake[0]
        if self.direction == Direction.UP:
            return (head[0], (head[1] - 1) % self.grid_size)
        elif self.direction == Direction.RIGHT:
            return ((head[0] + 1) % self.grid_size, head[1])
        elif self.direction == Direction.DOWN:
            return (head[0], (head[1] + 1) % self.grid_size)
        else:  # LEFT
            return ((head[0] - 1) % self.grid_size, head[1])
    
    def _is_collision(self, position: Tuple[int, int]) -> bool:
        """Check if position collides with snake body"""
        return position in self.snake[1:]
    
    def _generate_food(self) -> Tuple[int, int]:
        """Generate food at random position (not on snake)"""
        while True:
            food = (
                random.randint(0, self.grid_size - 1),
                random.randint(0, self.grid_size - 1)
            )
            if food not in self.snake:
                return food
    
    def _get_state(self) -> np.ndarray:
        """
        Get state representation for RL agent
        
        Returns:
            State array with:
            - Snake head position (normalized)
            - Food position (normalized)
            - Direction (one-hot encoded)
            - Danger zones (left, straight, right)
        """
        head = self.snake[0]
        
        # Normalize positions
        head_norm = [head[0] / self.grid_size, head[1] / self.grid_size]
        food_norm = [self.food[0] / self.grid_size, self.food[1] / self.grid_size]
        
        # Direction one-hot encoding
        direction_onehot = [0, 0, 0, 0]
        direction_onehot[self.direction.value] = 1
        
        # Danger zones (simplified)
        danger_left = 1.0 if self._is_dangerous(0) else 0.0
        danger_straight = 1.0 if self._is_dangerous(1) else 0.0
        danger_right = 1.0 if self._is_dangerous(2) else 0.0
        
        state = np.array(
            head_norm + food_norm + direction_onehot + 
            [danger_left, danger_straight, danger_right],
            dtype=np.float32
        )
        
        return state
    
    def _is_dangerous(self, action: int) -> bool:
        """Check if taking action would lead to collision"""
        # Temporarily change direction
        original_direction = self.direction
        self._update_direction(action)
        new_head = self._get_new_head()
        is_dangerous = self._is_collision(new_head)
        
        # Restore original direction
        self.direction = original_direction
        return is_dangerous
    
    def _render(self):
        """Render the game using Pygame"""
        self.screen.fill(self.BLACK)
        
        # Draw snake
        for segment in self.snake:
            x, y = segment
            pygame.draw.rect(
                self.screen, self.GREEN,
                (x * self.cell_size, y * self.cell_size, 
                 self.cell_size, self.cell_size)
            )
        
        # Draw food
        x, y = self.food
        pygame.draw.rect(
            self.screen, self.RED,
            (x * self.cell_size, y * self.cell_size, 
             self.cell_size, self.cell_size)
        )
        
        # Draw grid lines
        for i in range(self.grid_size + 1):
            pygame.draw.line(
                self.screen, self.WHITE,
                (i * self.cell_size, 0),
                (i * self.cell_size, self.grid_size * self.cell_size)
            )
            pygame.draw.line(
                self.screen, self.WHITE,
                (0, i * self.cell_size),
                (self.grid_size * self.cell_size, i * self.cell_size)
            )
        
        pygame.display.flip()
        self.clock.tick(self.fps)
    
    def get_action_space(self) -> int:
        """Get number of possible actions"""
        return 3  # Left, Straight, Right
    
    def get_state_space(self) -> int:
        """Get state space dimension"""
        return 11  # 2 (head) + 2 (food) + 4 (direction) + 3 (danger)
    
    def close(self):
        """Close the environment"""
        if self.render_mode == "human":
            pygame.quit()
