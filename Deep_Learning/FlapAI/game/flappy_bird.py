import pygame
import random
import math
from typing import List, Tuple, Optional, Dict, Any
import os

class Pipe:
    """Represents a pipe obstacle in the game."""
    
    def __init__(self, x: float, gap_y: float, gap_size: int = 150):
        self.x = x
        self.gap_y = gap_y
        self.gap_size = gap_size
        self.width = 50
        self.passed = False
        
    def update(self, speed: float) -> None:
        """Update pipe position."""
        self.x -= speed
        
    def draw(self, screen: pygame.Surface) -> None:
        """Draw the pipe on screen."""
        # Top pipe
        pygame.draw.rect(screen, (0, 255, 0), 
                        (self.x, 0, self.width, self.gap_y))
        # Bottom pipe
        pygame.draw.rect(screen, (0, 255, 0), 
                        (self.x, self.gap_y + self.gap_size, 
                         self.width, 600 - (self.gap_y + self.gap_size)))
        
    def get_rects(self) -> List[pygame.Rect]:
        """Get collision rectangles for the pipe."""
        top_rect = pygame.Rect(self.x, 0, self.width, self.gap_y)
        bottom_rect = pygame.Rect(self.x, self.gap_y + self.gap_size, 
                                 self.width, 600 - (self.gap_y + self.gap_size))
        return [top_rect, bottom_rect]
        
    def is_off_screen(self) -> bool:
        """Check if pipe is off screen."""
        return self.x + self.width < 0

class Bird:
    """Represents the bird character in the game."""
    
    def __init__(self, x: float = 100, y: float = 300):
        self.x = x
        self.y = y
        self.velocity = 0
        self.gravity = 0.5
        self.jump_strength = -8
        self.size = 20
        self.alive = True
        
    def jump(self) -> None:
        """Make the bird jump."""
        if self.alive:
            self.velocity = self.jump_strength
            
    def update(self) -> None:
        """Update bird physics."""
        if self.alive:
            self.velocity += self.gravity
            self.y += self.velocity
            
            # Check boundaries
            if self.y < 0:
                self.y = 0
                self.velocity = 0
            elif self.y > 580:
                self.alive = False
                
    def draw(self, screen: pygame.Surface) -> None:
        """Draw the bird on screen."""
        color = (255, 255, 0) if self.alive else (255, 0, 0)
        pygame.draw.circle(screen, color, (int(self.x), int(self.y)), self.size)
        
    def get_rect(self) -> pygame.Rect:
        """Get collision rectangle for the bird."""
        return pygame.Rect(self.x - self.size, self.y - self.size, 
                          self.size * 2, self.size * 2)
        
    def get_state(self) -> Dict[str, float]:
        """Get current state of the bird for AI agents."""
        return {
            'x': self.x,
            'y': self.y,
            'velocity': self.velocity,
            'alive': float(self.alive)
        }

class FlappyBirdGame:
    """Main game class for Flappy Bird."""
    
    def __init__(self, width: int = 800, height: int = 600, 
                 fps: int = 60, headless: bool = False):
        self.width = width
        self.height = height
        self.fps = fps
        self.headless = headless
        
        # Game state
        self.score = 0
        self.game_speed = 3
        self.pipe_spawn_timer = 0
        self.pipe_spawn_delay = 150  # frames between pipe spawns
        
        # Game objects
        self.bird = Bird()
        self.pipes: List[Pipe] = []
        
        # Pygame setup
        if not self.headless:
            pygame.init()
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("FlapAI - Flappy Bird")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)
        
        # AI state
        self.frame_count = 0
        self.last_action = 0
        
    def reset(self) -> Dict[str, Any]:
        """Reset the game to initial state."""
        self.score = 0
        self.game_speed = 3
        self.pipe_spawn_timer = 0
        self.frame_count = 0
        self.last_action = 0
        
        self.bird = Bird()
        self.pipes = []
        
        return self.get_state()
        
    def step(self, action: int = None) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Take one step in the game.
        
        Args:
            action: 0 for no flap, 1 for flap
            
        Returns:
            (state, reward, done, info)
        """
        # Handle action
        if action == 1:
            self.bird.jump()
            self.last_action = 1
        else:
            self.last_action = 0
            
        # Update game objects
        self.bird.update()
        
        # Spawn pipes
        self.pipe_spawn_timer += 1
        if self.pipe_spawn_timer >= self.pipe_spawn_delay:
            gap_y = random.randint(100, 400)
            self.pipes.append(Pipe(self.width, gap_y))
            self.pipe_spawn_timer = 0
            
        # Update pipes
        for pipe in self.pipes[:]:
            pipe.update(self.game_speed)
            
            # Check if bird passed pipe
            if not pipe.passed and pipe.x + pipe.width < self.bird.x:
                pipe.passed = True
                self.score += 1
                
            # Remove off-screen pipes
            if pipe.is_off_screen():
                self.pipes.remove(pipe)
                
        # Check collisions
        reward = 0
        done = False
        
        if self.bird.alive:
            # Check pipe collisions
            bird_rect = self.bird.get_rect()
            for pipe in self.pipes:
                for pipe_rect in pipe.get_rects():
                    if bird_rect.colliderect(pipe_rect):
                        self.bird.alive = False
                        reward = -100
                        done = True
                        break
                if done:
                    break
                    
            # Reward for staying alive
            if not done:
                reward = 1
                
        else:
            done = True
            reward = -100
            
        # Update frame count
        self.frame_count += 1
        
        return self.get_state(), reward, done, {'score': self.score}
        
    def get_state(self) -> Dict[str, Any]:
        """Get current game state for AI agents."""
        # Find closest pipe
        closest_pipe = None
        min_distance = float('inf')
        
        for pipe in self.pipes:
            if pipe.x + pipe.width > self.bird.x:
                distance = pipe.x - self.bird.x
                if distance < min_distance:
                    min_distance = distance
                    closest_pipe = pipe
                    
        # State vector for AI
        state = {
            'bird_y': self.bird.y / self.height,  # Normalized
            'bird_velocity': self.bird.velocity / 10.0,  # Normalized
            'bird_alive': float(self.bird.alive),
            'score': self.score,
            'frame_count': self.frame_count
        }
        
        if closest_pipe:
            state.update({
                'pipe_x': closest_pipe.x / self.width,  # Normalized
                'pipe_gap_y': closest_pipe.gap_y / self.height,  # Normalized
                'pipe_gap_size': closest_pipe.gap_size / self.height,  # Normalized
                'distance_to_pipe': min_distance / self.width  # Normalized
            })
        else:
            # No pipes on screen
            state.update({
                'pipe_x': 1.0,
                'pipe_gap_y': 0.5,
                'pipe_gap_size': 0.25,
                'distance_to_pipe': 1.0
            })
            
        return state
        
    def render(self) -> None:
        """Render the game (only if not headless)."""
        if self.headless:
            return
            
        # Clear screen
        self.screen.fill((135, 206, 235))  # Sky blue
        
        # Draw pipes
        for pipe in self.pipes:
            pipe.draw(self.screen)
            
        # Draw bird
        self.bird.draw(self.screen)
        
        # Draw score
        score_text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))
        
        # Update display
        pygame.display.flip()
        self.clock.tick(self.fps)
        
    def handle_events(self) -> bool:
        """Handle pygame events. Returns True if game should continue."""
        if self.headless:
            return True
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.bird.jump()
                elif event.key == pygame.K_r:
                    self.reset()
                    
        return True
        
    def run_human_game(self) -> int:
        """Run the game for human play. Returns final score."""
        running = True
        
        while running:
            running = self.handle_events()
            if not running:
                break
                
            # Game step
            state, reward, done, info = self.step()
            
            # Render
            self.render()
            
            if done:
                print(f"Game Over! Score: {self.score}")
                break
                
        return self.score
        
    def close(self) -> None:
        """Clean up pygame resources."""
        if not self.headless:
            pygame.quit()

def main():
    """Main function to run the game."""
    game = FlappyBirdGame()
    try:
        final_score = game.run_human_game()
        print(f"Final Score: {final_score}")
    finally:
        game.close()

if __name__ == "__main__":
    main() 