# Importing necessary libraries
import pygame
import time
import random

# Game speed
snake_speed = 15

# Window dimensions
window_x = 720
window_y = 480

# Color definitions
black = pygame.Color(0, 0, 0)
white = pygame.Color(255, 255, 255)
red = pygame.Color(255, 0, 0)
green = pygame.Color(0, 255, 0)
blue = pygame.Color(0, 0, 255)

# Initialize Pygame
pygame.init()

# Setup game window
pygame.display.set_caption('Snake Game')
game_window = pygame.display.set_mode((window_x, window_y))

# Frame rate controller
fps = pygame.time.Clock()

# Initialize snake's position and body
snake_position = [100, 50]
snake_body = [[100, 50], [90, 50], [80, 50], [70, 50]]

# Generate fruit position
fruit_position = [random.randrange(1, (window_x // 10)) * 10,
                  random.randrange(1, (window_y // 10)) * 10]

fruit_spawn = True

# Default direction
direction = 'RIGHT'
change_to = direction

# Initial score
score = 0

# Function to display score
def show_score(choice, color, font, size):
    score_font = pygame.font.SysFont(font, size)
    score_surface = score_font.render('Score : ' + str(score), True, color)
    score_rect = score_surface.get_rect()
    game_window.blit(score_surface, score_rect)

# Function for game over scenario
def game_over():
    my_font = pygame.font.SysFont('times new roman', 50)
    game_over_surface = my_font.render('Your Score is : ' + str(score), True, red)
    game_over_rect = game_over_surface.get_rect()
    game_over_rect.midtop = (window_x / 2, window_y / 4)
    game_window.blit(game_over_surface, game_over_rect)
    pygame.display.flip()
    time.sleep(2)
    pygame.quit()
    quit()

# Main game loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP: change_to = 'UP'
            if event.key == pygame.K_DOWN: change_to = 'DOWN'
            if event.key == pygame.K_LEFT: change_to = 'LEFT'
            if event.key == pygame.K_RIGHT: change_to = 'RIGHT'

    # Prevent snake from moving in opposite direction
    if change_to == 'UP' and direction != 'DOWN': direction = 'UP'
    if change_to == 'DOWN' and direction != 'UP': direction = 'DOWN'
    if change_to == 'LEFT' and direction != 'RIGHT': direction = 'LEFT'
    if change_to == 'RIGHT' and direction != 'LEFT': direction = 'RIGHT'

    # Move the snake
    if direction == 'UP': snake_position[1] -= 10
    if direction == 'DOWN': snake_position[1] += 10
    if direction == 'LEFT': snake_position[0] -= 10
    if direction == 'RIGHT': snake_position[0] += 10

    # Snake grows when eating fruit
    snake_body.insert(0, list(snake_position))
    if snake_position == fruit_position:
        score += 10
        fruit_spawn = False
    else:
        snake_body.pop()

    # Generate new fruit
    if not fruit_spawn:
        fruit_position = [random.randrange(1, (window_x // 10)) * 10,
                          random.randrange(1, (window_y // 10)) * 10]
    fruit_spawn = True

    # Clear the game window
    game_window.fill(black)

    # Draw the snake and fruit
    for pos in snake_body:
        pygame.draw.rect(game_window, green, pygame.Rect(pos[0], pos[1], 10, 10))
    pygame.draw.rect(game_window, white, pygame.Rect(fruit_position[0], fruit_position[1], 10, 10))

    # Game over conditions
    if snake_position[0] < 0 or snake_position[0] > window_x - 10 or \
       snake_position[1] < 0 or snake_position[1] > window_y - 10:
        game_over()

    # Check for self-collision
    for block in snake_body[1:]:
        if snake_position == block: game_over()

    # Display score
    show_score(1, white, 'times new roman', 20)

    # Refresh the game screen
    pygame.display.update()

    # Control game speed
    fps.tick(snake_speed)
