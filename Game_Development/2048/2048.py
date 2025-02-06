import pygame
import random
import os
import json

# Constants
GRID_SIZE = 4
CELL_SIZE = 100
GRID_COLOR = (187, 173, 160)
CELL_COLORS = {
    0: (205, 193, 180),
    2: (238, 228, 218),
    4: (237, 224, 200),
    8: (242, 177, 121),
    16: (245, 149, 99),
    32: (246, 124, 95),
    64: (246, 94, 59),
    128: (237, 207, 114),
    256: (237, 204, 97),
    512: (237, 200, 80),
    1024: (237, 197, 63),
    2048: (237, 194, 46),
}

# Leaderboard file
LEADERBOARD_FILE = "leaderboard.json"

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE + 150))  # Extra space for UI elements
pygame.display.set_caption('2048 Game')
font = pygame.font.Font(None, 40)
score_font = pygame.font.Font(None, 36)

# Game Functions
def load_leaderboard():
    if os.path.exists(LEADERBOARD_FILE):
        with open(LEADERBOARD_FILE, "r") as file:
            return json.load(file)
    return {}

def save_leaderboard(leaderboard):
    with open(LEADERBOARD_FILE, "w") as file:
        json.dump(leaderboard, file, indent=4)

def init_game():
    grid = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]
    add_random_tile(grid)
    add_random_tile(grid)
    return grid

def add_random_tile(grid):
    empty_cells = [(i, j) for i in range(GRID_SIZE) for j in range(GRID_SIZE) if grid[i][j] == 0]
    if empty_cells:
        x, y = random.choice(empty_cells)
        grid[x][y] = random.choice([2, 4])

def draw_grid(grid, score, high_score, username):
    # Draw the grid cells
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            value = grid[x][y]
            color = CELL_COLORS.get(value, (60, 58, 50))
            pygame.draw.rect(screen, color, (y * CELL_SIZE, x * CELL_SIZE + 100, CELL_SIZE, CELL_SIZE))  # Shifted down for UI space
            if value != 0:
                text = font.render(str(value), True, (255, 255, 255))
                text_rect = text.get_rect(center=(y * CELL_SIZE + CELL_SIZE // 2, x * CELL_SIZE + 100 + CELL_SIZE // 2))
                screen.blit(text, text_rect)
    
    # Draw the scores
    score_text = score_font.render(f"Score: {score}", True, (255, 255, 255))
    screen.blit(score_text, (10, 10))

    high_score_text = score_font.render(f"High Score: {high_score}", True, (255, 255, 255))
    screen.blit(high_score_text, (10, 50))

    # Draw the username
    username_text = score_font.render(f"Player: {username}", True, (255, 255, 255))
    screen.blit(username_text, (10, 90))

def move_left(grid):
    new_grid = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]
    for x in range(GRID_SIZE):
        position = 0
        for y in range(GRID_SIZE):
            if grid[x][y] != 0:
                if new_grid[x][position] == 0:
                    new_grid[x][position] = grid[x][y]
                elif new_grid[x][position] == grid[x][y]:
                    new_grid[x][position] *= 2
                    position += 1
                else:
                    position += 1
                    new_grid[x][position] = grid[x][y]
    return new_grid

def move_right(grid):
    new_grid = [row[::-1] for row in grid]
    new_grid = move_left(new_grid)
    return [row[::-1] for row in new_grid]

def move_up(grid):
    new_grid = [[grid[y][x] for y in range(GRID_SIZE)] for x in range(GRID_SIZE)]
    new_grid = move_left(new_grid)
    return [[new_grid[y][x] for y in range(GRID_SIZE)] for x in range(GRID_SIZE)]

def move_down(grid):
    new_grid = [[grid[y][x] for y in range(GRID_SIZE)] for x in range(GRID_SIZE)]
    new_grid = move_right(new_grid)
    return [[new_grid[y][x] for y in range(GRID_SIZE)] for x in range(GRID_SIZE)]

def check_win(grid):
    for row in grid:
        if 2048 in row:
            return True
    return False

def check_game_over(grid):
    for row in grid:
        if 0 in row:
            return False

    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE - 1):
            if grid[x][y] == grid[x][y + 1] or grid[y][x] == grid[y + 1][x]:
                return False

    return True

# Main Game Loop
def game_loop(username):
    grid = init_game()
    current_score = 0
    leaderboard = load_leaderboard()
    game_over = False
    game_won = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

            if event.type == pygame.KEYDOWN and not game_over and not game_won:
                if event.key == pygame.K_LEFT:
                    grid = move_left(grid)
                elif event.key == pygame.K_RIGHT:
                    grid = move_right(grid)
                elif event.key == pygame.K_UP:
                    grid = move_up(grid)
                elif event.key == pygame.K_DOWN:
                    grid = move_down(grid)
                add_random_tile(grid)

                # Update the current score based on merged tiles (this is where you'd add your logic)
                current_score += 10  # Example, replace with actual score logic

        screen.fill(GRID_COLOR)
        draw_grid(grid, current_score, current_score, username)

        # Check for win
        if check_win(grid):
            game_won = True
            pygame.display.set_caption('You Won!')

        # Check for game over
        if check_game_over(grid):
            game_over = True
            pygame.display.set_caption('Game Over!')

        # Update the leaderboard if needed
        if current_score > leaderboard.get(username, 0):
            leaderboard[username] = current_score
            save_leaderboard(leaderboard)

        pygame.display.flip()

        if game_over or game_won:
            pygame.time.wait(2000)
            return leaderboard

# Display the leaderboard
def display_leaderboard():
    leaderboard = load_leaderboard()
    leaderboard_sorted = sorted(leaderboard.items(), key=lambda x: x[1], reverse=True)

    leaderboard_text = "Leaderboard:\n"
    for idx, (user, score) in enumerate(leaderboard_sorted[:5]):  # Display top 5
        leaderboard_text += f"{idx + 1}. {user}: {score}\n"

    leaderboard_surface = pygame.Surface((GRID_SIZE * CELL_SIZE, 150))
    leaderboard_surface.fill((0, 0, 0))
    leaderboard_surface.set_alpha(200)
    screen.blit(leaderboard_surface, (0, GRID_SIZE * CELL_SIZE))

    leaderboard_display = score_font.render(leaderboard_text, True, (255, 255, 255))
    screen.blit(leaderboard_display, (10, GRID_SIZE * CELL_SIZE + 10))

    pygame.display.flip()
    pygame.time.wait(5000)

# Main Menu - Allow the user to input their username
def main_menu():
    username = ''
    input_box = pygame.Rect(100, 100, 140, 32)
    color_inactive = pygame.Color('lightskyblue3')
    color_active = pygame.Color('dodgerblue2')
    color = color_inactive
    active = False
    text = ''
    clock = pygame.time.Clock()

    while True:
        screen.fill((30, 30, 30))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                if input_box.collidepoint(event.pos):
                    active = True
                    color = color_active
                else:
                    active = False
                    color = color_inactive

            if event.type == pygame.KEYDOWN:
                if active:
                    if event.key == pygame.K_RETURN:
                        return text
                    elif event.key == pygame.K_BACKSPACE:
                        text = text[:-1]
                    else:
                        text += event.unicode

        txt_surface = font.render(text, True, color)
        width = max(200, txt_surface.get_width()+10)
        input_box.w = width
        screen.blit(txt_surface, (input_box.x+5, input_box.y+5))
        pygame.draw.rect(screen, color, input_box, 2)

        pygame.display.flip()
        clock.tick(30)

# Start Game
username = main_menu()
leaderboard = game_loop(username)
display_leaderboard()
pygame.quit()
