import pygame
import random
import sys

# Initialize pygame
pygame.init()

# Screen dimensions and setup
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 400
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Snake Game")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

# Game variables
snake_size = 20
snake_speed = 15
snake_pos = [100, 50]
snake_body = [[100, 50], [80, 50], [60, 50]]
direction = 'RIGHT'
change_to = direction
score = 0

# Food
food_pos = [random.randrange(1, (SCREEN_WIDTH // snake_size)) * snake_size,
            random.randrange(1, (SCREEN_HEIGHT // snake_size)) * snake_size]
food_spawn = True

# Fonts
font = pygame.font.SysFont('Arial', 24)

# Game over function
def game_over():
    screen.fill(BLACK)
    game_over_text = font.render(f"Game Over! Your score: {score}", True, WHITE)
    screen.blit(game_over_text, [SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT // 2])
    pygame.display.flip()
    pygame.time.sleep(2)
    pygame.quit()
    sys.exit()

# Main game loop
clock = pygame.time.Clock()
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP and direction != 'DOWN':
                change_to = 'UP'
            elif event.key == pygame.K_DOWN and direction != 'UP':
                change_to = 'DOWN'
            elif event.key == pygame.K_LEFT and direction != 'RIGHT':
                change_to = 'LEFT'
            elif event.key == pygame.K_RIGHT and direction != 'LEFT':
                change_to = 'RIGHT'

    # Update direction
    direction = change_to

    # Move the snake
    if direction == 'UP':
        snake_pos[1] -= snake_size
    elif direction == 'DOWN':
        snake_pos[1] += snake_size
    elif direction == 'LEFT':
        snake_pos[0] -= snake_size
    elif direction == 'RIGHT':
        snake_pos[0] += snake_size

    # Snake body growing mechanism
    snake_body.insert(0, list(snake_pos))
    if snake_pos == food_pos:
        score += 10
        food_spawn = False
    else:
        snake_body.pop()

    if not food_spawn:
        food_pos = [random.randrange(1, (SCREEN_WIDTH // snake_size)) * snake_size,
                    random.randrange(1, (SCREEN_HEIGHT // snake_size)) * snake_size]
    food_spawn = True

    # Game Over conditions
    if (snake_pos[0] < 0 or snake_pos[0] > SCREEN_WIDTH - snake_size or
            snake_pos[1] < 0 or snake_pos[1] > SCREEN_HEIGHT - snake_size):
        game_over()
    
    for block in snake_body[1:]:
        if snake_pos == block:
            game_over()

    # Fill screen and draw snake and food
    screen.fill(BLACK)
    for pos in snake_body:
        pygame.draw.rect(screen, GREEN, pygame.Rect(pos[0], pos[1], snake_size, snake_size))

    pygame.draw.rect(screen, RED, pygame.Rect(food_pos[0], food_pos[1], snake_size, snake_size))

    # Display score
    score_text = font.render(f"Score: {score}", True, WHITE)
    screen.blit(score_text, [0, 0])

    # Refresh game screen and set FPS
    pygame.display.update()
    clock.tick(snake_speed)
