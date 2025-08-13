import sys
import os
import pytest
import pygame
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.flappy_bird import FlappyBirdGame, Bird, Pipe

class TestBird:
    """Test cases for the Bird class."""
    
    def test_bird_initialization(self):
        """Test bird initialization with default values."""
        bird = Bird()
        assert bird.x == 100
        assert bird.y == 300
        assert bird.velocity == 0
        assert bird.alive == True
        assert bird.size == 20
        
    def test_bird_initialization_custom(self):
        """Test bird initialization with custom values."""
        bird = Bird(x=50, y=200)
        assert bird.x == 50
        assert bird.y == 200
        assert bird.alive == True
        
    def test_bird_jump(self):
        """Test bird jump mechanics."""
        bird = Bird()
        initial_velocity = bird.velocity
        bird.jump()
        assert bird.velocity == bird.jump_strength
        assert bird.velocity < 0  # Should be negative (upward)
        
    def test_bird_jump_when_dead(self):
        """Test that dead birds cannot jump."""
        bird = Bird()
        bird.alive = False
        initial_velocity = bird.velocity
        bird.jump()
        assert bird.velocity == initial_velocity  # Should not change
        
    def test_bird_update_physics(self):
        """Test bird physics update."""
        bird = Bird()
        initial_y = bird.y
        initial_velocity = bird.velocity
        
        bird.update()
        
        # Should fall due to gravity
        assert bird.y > initial_y
        assert bird.velocity > initial_velocity
        
    def test_bird_boundary_collision_top(self):
        """Test bird collision with top boundary."""
        bird = Bird(y=0)
        bird.velocity = -10  # Moving upward
        bird.update()
        
        assert bird.y == 0  # Should stop at top
        assert bird.velocity == 0  # Should stop moving
        
    def test_bird_boundary_collision_bottom(self):
        """Test bird collision with bottom boundary."""
        bird = Bird(y=580)
        bird.velocity = 10  # Moving downward
        bird.update()
        
        assert bird.alive == False  # Should die when hitting bottom
        
    def test_bird_get_rect(self):
        """Test bird collision rectangle."""
        bird = Bird(x=100, y=200)
        rect = bird.get_rect()
        
        assert rect.x == 80  # x - size
        assert rect.y == 180  # y - size
        assert rect.width == 40  # size * 2
        assert rect.height == 40  # size * 2
        
    def test_bird_get_state(self):
        """Test bird state dictionary."""
        bird = Bird(x=100, y=200)
        state = bird.get_state()
        
        assert state['x'] == 100
        assert state['y'] == 200
        assert state['velocity'] == 0
        assert state['alive'] == 1.0

class TestPipe:
    """Test cases for the Pipe class."""
    
    def test_pipe_initialization(self):
        """Test pipe initialization."""
        pipe = Pipe(x=400, gap_y=200)
        assert pipe.x == 400
        assert pipe.gap_y == 200
        assert pipe.gap_size == 150
        assert pipe.width == 50
        assert pipe.passed == False
        
    def test_pipe_initialization_custom(self):
        """Test pipe initialization with custom gap size."""
        pipe = Pipe(x=400, gap_y=200, gap_size=100)
        assert pipe.gap_size == 100
        
    def test_pipe_update(self):
        """Test pipe movement."""
        pipe = Pipe(x=400, gap_y=200)
        initial_x = pipe.x
        speed = 3
        
        pipe.update(speed)
        
        assert pipe.x == initial_x - speed
        
    def test_pipe_get_rects(self):
        """Test pipe collision rectangles."""
        pipe = Pipe(x=400, gap_y=200, gap_size=100)
        rects = pipe.get_rects()
        
        assert len(rects) == 2  # Top and bottom pipe
        
        # Top pipe
        top_rect = rects[0]
        assert top_rect.x == 400
        assert top_rect.y == 0
        assert top_rect.width == 50
        assert top_rect.height == 200
        
        # Bottom pipe
        bottom_rect = rects[1]
        assert bottom_rect.x == 400
        assert bottom_rect.y == 300  # gap_y + gap_size
        assert bottom_rect.width == 50
        assert bottom_rect.height == 300  # 600 - (gap_y + gap_size)
        
    def test_pipe_is_off_screen(self):
        """Test pipe off-screen detection."""
        pipe = Pipe(x=400, gap_y=200)
        assert pipe.is_off_screen() == False
        
        pipe.x = -60  # x + width < 0
        assert pipe.is_off_screen() == True

class TestFlappyBirdGame:
    """Test cases for the FlappyBirdGame class."""
    
    def test_game_initialization(self):
        """Test game initialization."""
        game = FlappyBirdGame()
        assert game.width == 800
        assert game.height == 600
        assert game.score == 0
        assert game.game_speed == 3
        assert len(game.pipes) == 0
        assert game.bird.alive == True
        
    def test_game_initialization_headless(self):
        """Test game initialization in headless mode."""
        game = FlappyBirdGame(headless=True)
        assert game.headless == True
        assert game.width == 800
        assert game.height == 600
        
    def test_game_reset(self):
        """Test game reset functionality."""
        game = FlappyBirdGame()
        
        # Modify game state
        game.score = 10
        game.bird.y = 500
        game.pipes.append(Pipe(400, 200))
        
        # Reset game
        state = game.reset()
        
        assert game.score == 0
        assert game.bird.y == 300  # Reset to initial position
        assert len(game.pipes) == 0
        assert isinstance(state, dict)
        
    def test_game_step_no_action(self):
        """Test game step with no action."""
        game = FlappyBirdGame(headless=True)
        initial_state = game.reset()
        
        state, reward, done, info = game.step(0)
        
        assert isinstance(state, dict)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        assert 'score' in info
        
    def test_game_step_with_action(self):
        """Test game step with flap action."""
        game = FlappyBirdGame(headless=True)
        initial_state = game.reset()
        
        state, reward, done, info = game.step(1)
        
        assert isinstance(state, dict)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        
    def test_game_pipe_spawning(self):
        """Test pipe spawning mechanism."""
        game = FlappyBirdGame(headless=True)
        game.reset()
        
        # Advance time to spawn pipes
        for _ in range(game.pipe_spawn_delay + 1):
            game.step(0)
            
        assert len(game.pipes) > 0
        
    def test_game_pipe_passing(self):
        """Test pipe passing and scoring."""
        game = FlappyBirdGame(headless=True)
        game.reset()
        
        # Create a pipe that the bird can pass
        pipe = Pipe(x=90, gap_y=250)  # Bird at x=100, pipe at x=90
        game.pipes.append(pipe)
        
        # Move bird past pipe
        game.bird.x = 150
        game.step(0)
        
        assert pipe.passed == True
        assert game.score == 1
        
    def test_game_collision_detection(self):
        """Test collision detection between bird and pipes."""
        game = FlappyBirdGame(headless=True)
        game.reset()
        
        # Create a pipe that the bird will hit
        pipe = Pipe(x=80, gap_y=100)  # Bird at x=100, pipe at x=80
        game.pipes.append(pipe)
        
        # Position bird to hit the pipe
        game.bird.x = 100
        game.bird.y = 50  # Hit top pipe
        
        state, reward, done, info = game.step(0)
        
        assert game.bird.alive == False
        assert done == True
        assert reward == -100
        
    def test_game_boundary_collision(self):
        """Test bird collision with game boundaries."""
        game = FlappyBirdGame(headless=True)
        game.reset()
        
        # Move bird to bottom boundary
        game.bird.y = 580
        game.bird.velocity = 10
        
        state, reward, done, info = game.step(0)
        
        assert game.bird.alive == False
        assert done == True
        assert reward == -100
        
    def test_game_get_state(self):
        """Test game state retrieval."""
        game = FlappyBirdGame(headless=True)
        state = game.reset()
        
        assert isinstance(state, dict)
        assert 'bird_y' in state
        assert 'bird_velocity' in state
        assert 'bird_alive' in state
        assert 'score' in state
        assert 'frame_count' in state
        assert 'pipe_x' in state
        assert 'pipe_gap_y' in state
        assert 'pipe_gap_size' in state
        assert 'distance_to_pipe' in state
        
        # Check normalization
        assert 0 <= state['bird_y'] <= 1
        assert 0 <= state['pipe_x'] <= 1
        assert 0 <= state['pipe_gap_y'] <= 1
        
    def test_game_state_with_pipes(self):
        """Test game state when pipes are present."""
        game = FlappyBirdGame(headless=True)
        game.reset()
        
        # Add a pipe
        pipe = Pipe(x=200, gap_y=250)
        game.pipes.append(pipe)
        
        state = game.get_state()
        
        assert state['pipe_x'] == 200 / game.width
        assert state['pipe_gap_y'] == 250 / game.height
        assert state['pipe_gap_size'] == 150 / game.height
        
    def test_game_state_without_pipes(self):
        """Test game state when no pipes are present."""
        game = FlappyBirdGame(headless=True)
        game.reset()
        
        state = game.get_state()
        
        # Should have default values when no pipes
        assert state['pipe_x'] == 1.0
        assert state['pipe_gap_y'] == 0.5
        assert state['pipe_gap_size'] == 0.25
        assert state['distance_to_pipe'] == 1.0
        
    def test_game_frame_count(self):
        """Test frame count increment."""
        game = FlappyBirdGame(headless=True)
        game.reset()
        
        initial_frame = game.frame_count
        game.step(0)
        
        assert game.frame_count == initial_frame + 1
        
    def test_game_reward_system(self):
        """Test reward system."""
        game = FlappyBirdGame(headless=True)
        game.reset()
        
        # Test reward for staying alive
        state, reward, done, info = game.step(0)
        assert reward == 1  # Reward for staying alive
        
        # Test reward for dying
        game.bird.alive = False
        state, reward, done, info = game.step(0)
        assert reward == -100  # Penalty for dying

class TestGameIntegration:
    """Integration tests for the complete game."""
    
    def test_complete_game_episode(self):
        """Test a complete game episode."""
        game = FlappyBirdGame(headless=True)
        state = game.reset()
        
        total_reward = 0
        steps = 0
        done = False
        
        while not done and steps < 1000:  # Prevent infinite loop
            action = 0  # No flap
            state, reward, done, info = game.step(action)
            total_reward += reward
            steps += 1
            
        assert steps > 0
        assert done == True
        assert game.bird.alive == False
        
    def test_game_with_random_actions(self):
        """Test game with random actions."""
        game = FlappyBirdGame(headless=True)
        state = game.reset()
        
        total_reward = 0
        steps = 0
        done = False
        
        while not done and steps < 1000:
            action = np.random.randint(0, 2)  # Random action
            state, reward, done, info = game.step(action)
            total_reward += reward
            steps += 1
            
        assert steps > 0
        assert done == True
        
    def test_game_performance(self):
        """Test game performance (should be fast)."""
        import time
        
        game = FlappyBirdGame(headless=True)
        game.reset()
        
        start_time = time.time()
        
        for _ in range(1000):
            game.step(0)
            
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Should complete 1000 steps in reasonable time
        assert elapsed_time < 10.0  # Less than 10 seconds
        
    def test_game_consistency(self):
        """Test game consistency across multiple runs."""
        game = FlappyBirdGame(headless=True)
        
        # Run multiple episodes and check consistency
        scores = []
        for _ in range(5):
            game.reset()
            total_reward = 0
            done = False
            
            while not done:
                state, reward, done, info = game.step(0)
                total_reward += reward
                
            scores.append(info['score'])
            
        # All episodes should complete
        assert len(scores) == 5
        assert all(isinstance(score, int) for score in scores)

if __name__ == "__main__":
    pytest.main([__file__]) 