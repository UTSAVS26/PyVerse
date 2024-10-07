import pygame
import time
import random

# Initialize pygame
pygame.init()

# Define colors
white = (255, 255, 255)
yellow = (255, 255, 102)
black = (0, 0, 0)
red = (213, 50, 80)
green = (0, 255, 0)
blue = (50, 153, 213)

# Define display size
width = 600
height = 400

# Create the game window
game_display = pygame.display.set_mode((width, height))
pygame.display.set_caption('Snake Game')

# Define game clock
clock = pygame.time.Clock()

# Define snake block size and speed
snake_block = 10
snake_speed = 15

# Define font style and size
font_style = pygame.font.SysFont("bahnschrift", 25)
score_font = pygame.font.SysFont("comicsansms", 35)

# Function to display the player's score
def show_score(score):
    value = score_font.render("Your Score: " + str(score), True, yellow)
    game_display.blit(value, [0, 0])

# Function to draw the snake
def draw_snake(snake_block, snake_list):
    for x in snake_list:
        pygame.draw.rect(game_display, green, [x[0], x[1], snake_block, snake_block])

# Function to display messages
def message(msg, color):
    mesg = font_style.render(msg, True, color)
    game_display.blit(mesg, [width / 6, height / 3])

# Game loop
def game_loop():
    game_over = False
    game_close = False

    # Starting position of the snake
    x = width / 2
    y = height / 2

    # Snake movement
    x_change = 0
    y_change = 0

    # Snake body list and initial length
    snake_list = []
    length_of_snake = 1

    # Random food location
    food_x = round(random.randrange(0, width - snake_block) / 10.0) * 10.0
    food_y = round(random.randrange(0, height - snake_block) / 10.0) * 10.0

    while not game_over:

        while game_close:
            game_display.fill(blue)
            message("You lost! Press Q-Quit or C-Play Again", red)
            show_score(length_of_snake - 1)
            pygame.display.update()

            # Check for events in the lose state
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        game_over = True
                        game_close = False
                    if event.key == pygame.K_c:
                        game_loop()

        # Movement and boundary check
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    x_change = -snake_block
                    y_change = 0
                elif event.key == pygame.K_RIGHT:
                    x_change = snake_block
                    y_change = 0
                elif event.key == pygame.K_UP:
                    y_change = -snake_block
                    x_change = 0
                elif event.key == pygame.K_DOWN:
                    y_change = snake_block
                    x_change = 0

        # If the snake goes out of bounds, game over
        if x >= width or x < 0 or y >= height or y < 0:
            game_close = True
        x += x_change
        y += y_change
        game_display.fill(black)

        # Draw the food
        pygame.draw.rect(game_display, red, [food_x, food_y, snake_block, snake_block])

        # Add the new position to the snake body
        snake_head = []
        snake_head.append(x)
        snake_head.append(y)
        snake_list.append(snake_head)
        if len(snake_list) > length_of_snake:
            del snake_list[0]

        # If the snake collides with itself, game over
        for segment in snake_list[:-1]:
            if segment == snake_head:
                game_close = True

        draw_snake(snake_block, snake_list)
        show_score(length_of_snake - 1)

        pygame.display.update()

        # If the snake eats the food
        if x == food_x and y == food_y:
            food_x = round(random.randrange(0, width - snake_block) / 10.0) * 10.0
            food_y = round(random.randrange(0, height - snake_block) / 10.0) * 10.0
            length_of_snake += 1

        clock.tick(snake_speed)

    pygame.quit()
    quit()

# Start the game
game_loop()
