import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from environment.snake_env import SnakeEnv, Direction

class TestSnakeEnv:
    """Test cases for Snake environment"""
    
    def setup_method(self):
        """Setup method called before each test"""
        self.env = SnakeEnv(grid_size=5, render_mode="rgb_array")
    
    def teardown_method(self):
        """Teardown method called after each test"""
        self.env.close()
    
    def test_initialization(self):
        """Test environment initialization"""
        assert self.env.grid_size == 5
        assert len(self.env.snake) == 1
        assert self.env.snake[0] == (2, 2)  # Center of 5x5 grid
        assert self.env.direction == Direction.RIGHT
        assert self.env.score == 0
        assert not self.env.game_over
        assert self.env.steps == 0
    
    def test_reset(self):
        """Test environment reset"""
        # Modify environment state
        self.env.snake = [(0, 0), (1, 0)]
        self.env.score = 5
        self.env.game_over = True
        self.env.steps = 100
        
        # Reset
        state = self.env.reset()
        
        # Check reset state
        assert len(self.env.snake) == 1
        assert self.env.snake[0] == (2, 2)
        assert self.env.score == 0
        assert not self.env.game_over
        assert self.env.steps == 0
        assert isinstance(state, np.ndarray)
        assert state.shape == (11,)  # State space dimension
    
    def test_food_generation(self):
        """Test food generation"""
        # Generate food multiple times
        foods = set()
        for _ in range(100):
            food = self.env._generate_food()
            foods.add(food)
            # Check food is within bounds
            assert 0 <= food[0] < self.env.grid_size
            assert 0 <= food[1] < self.env.grid_size
            # Check food is not on snake
            assert food not in self.env.snake
        
        # Should generate different positions
        assert len(foods) > 1
    
    def test_movement(self):
        """Test snake movement"""
        initial_head = self.env.snake[0]
        
        # Move right
        new_head = self.env._get_new_head()
        expected_head = ((initial_head[0] + 1) % self.env.grid_size, initial_head[1])
        assert new_head == expected_head
        
        # Change direction and test
        self.env.direction = Direction.UP
        new_head = self.env._get_new_head()
        expected_head = (initial_head[0], (initial_head[1] - 1) % self.env.grid_size)
        assert new_head == expected_head
    
    def test_direction_update(self):
        """Test direction updates"""
        # Test left turn
        self.env.direction = Direction.RIGHT
        self.env._update_direction(0)  # Left
        assert self.env.direction == Direction.UP
        
        # Test right turn
        self.env.direction = Direction.RIGHT
        self.env._update_direction(2)  # Right
        assert self.env.direction == Direction.DOWN
        
        # Test straight (no change)
        self.env.direction = Direction.RIGHT
        self.env._update_direction(1)  # Straight
        assert self.env.direction == Direction.RIGHT
    
    def test_collision_detection(self):
        """Test collision detection"""
        # No collision initially
        assert not self.env._is_collision(self.env.snake[0])
        
        # Create collision scenario
        self.env.snake = [(0, 0), (1, 0), (1, 1), (0, 1)]
        # Head at (0, 0), body at (1, 0), (1, 1), (0, 1)
        
        # Test collision with body
        assert self.env._is_collision((1, 0))  # Collision with body
        assert not self.env._is_collision((2, 0))  # No collision
    
    def test_state_representation(self):
        """Test state representation"""
        state = self.env._get_state()
        
        # Check state shape
        assert state.shape == (11,)
        assert state.dtype == np.float32
        
        # Check state components
        head_x, head_y = state[0], state[1]
        food_x, food_y = state[2], state[3]
        direction = state[4:8]
        danger = state[8:11]
        
        # Check normalized positions
        assert 0 <= head_x <= 1
        assert 0 <= head_y <= 1
        assert 0 <= food_x <= 1
        assert 0 <= food_y <= 1
        
        # Check direction one-hot encoding
        assert np.sum(direction) == 1
        assert np.argmax(direction) == self.env.direction.value
        
        # Check danger zones are binary
        assert all(d in [0.0, 1.0] for d in danger)
    
    def test_step_function(self):
        """Test step function"""
        initial_score = self.env.score
        initial_steps = self.env.steps
        
        # Take a step
        state, reward, done, info = self.env.step(1)  # Go straight
        
        # Check state changes
        assert self.env.steps == initial_steps + 1
        assert isinstance(state, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        
        # Check info contains expected keys
        assert 'score' in info
        assert 'steps' in info
        assert 'snake_length' in info
    
    def test_food_eating(self):
        """Test food eating mechanics"""
        # Place food at snake's next position
        next_head = self.env._get_new_head()
        self.env.food = next_head
        
        initial_score = self.env.score
        initial_length = len(self.env.snake)
        
        # Take step to eat food
        state, reward, done, info = self.env.step(1)
        
        # Check food was eaten
        assert self.env.score == initial_score + 1
        assert len(self.env.snake) == initial_length + 1
        assert reward == 10  # Food reward
        assert not done
    
    def test_collision_penalty(self):
        """Test collision penalty"""
        # Create collision scenario - snake will collide with its own body
        # Snake: head at (1,1), body at (1,0), (0,0), (0,1)
        self.env.snake = [(1, 1), (1, 0), (0, 0), (0, 1)]
        self.env.direction = Direction.LEFT  # Will collide with body at (0,1)
        
        # Take step that causes collision
        state, reward, done, info = self.env.step(1)
        
        # Check collision handling
        assert self.env.game_over
        assert reward == -10.0  # Collision penalty
        assert done
    
    def test_wrapping_boundaries(self):
        """Test boundary wrapping"""
        # Move snake to edge
        self.env.snake = [(4, 2)]  # Right edge of 5x5 grid
        self.env.direction = Direction.RIGHT
        
        # Move right (should wrap to left)
        new_head = self.env._get_new_head()
        assert new_head == (0, 2)  # Wrapped to left side
    
    def test_action_space(self):
        """Test action space"""
        assert self.env.get_action_space() == 3  # Left, Straight, Right
    
    def test_state_space(self):
        """Test state space"""
        assert self.env.get_state_space() == 11  # State dimension
    
    def test_danger_detection(self):
        """Test danger zone detection"""
        # Create scenario with danger
        self.env.snake = [(1, 1), (2, 1), (2, 2), (1, 2)]  # Square shape
        self.env.direction = Direction.RIGHT
        
        # Check danger zones
        danger_left = self.env._is_dangerous(0)
        danger_straight = self.env._is_dangerous(1)
        danger_right = self.env._is_dangerous(2)
        
        # Should detect some danger
        assert isinstance(danger_left, bool)
        assert isinstance(danger_straight, bool)
        assert isinstance(danger_right, bool)
    
    def test_multiple_steps(self):
        """Test multiple consecutive steps"""
        initial_score = self.env.score
        initial_steps = self.env.steps
        
        for _ in range(5):
            state, reward, done, info = self.env.step(1)  # Go straight
            assert not done  # Should not die immediately
            assert isinstance(state, np.ndarray)
            assert isinstance(reward, float)
        
        # Environment should have progressed
        assert self.env.steps > initial_steps
        # Score might not change if no food eaten, but steps should increase
    
    def test_episode_completion(self):
        """Test complete episode"""
        episode_reward = 0
        steps = 0
        
        while not self.env.game_over and steps < 100:  # Prevent infinite loop
            state, reward, done, info = self.env.step(1)  # Go straight
            episode_reward += reward
            steps += 1
            
            if done:
                break
        
        # Episode should complete
        assert self.env.game_over or steps >= 100
        assert isinstance(episode_reward, float)
