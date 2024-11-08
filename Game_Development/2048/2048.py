import pygame
import random

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

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE))
pygame.display.set_caption('2048 Game')
font = pygame.font.Font(None, 40)

# Game Functions
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

def draw_grid(grid):
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            value = grid[x][y]
            color = CELL_COLORS.get(value, (60, 58, 50))
            pygame.draw.rect(screen, color, (y * CELL_SIZE, x * CELL_SIZE, CELL_SIZE, CELL_SIZE))
            if value != 0:
                text = font.render(str(value), True, (255, 255, 255))
                text_rect = text.get_rect(center=(y * CELL_SIZE + CELL_SIZE // 2, x * CELL_SIZE + CELL_SIZE // 2))
                screen.blit(text, text_rect)

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
grid = init_game()
running = True
game_over = False
game_won = False

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
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

    screen.fill(GRID_COLOR)
    draw_grid(grid)
    
    # Check for win
    if check_win(grid):
        game_won = True
        pygame.display.set_caption('You Won!')

    # Check for game over
    if check_game_over(grid):
        game_over = True
        pygame.display.set_caption('Game Over!')

    pygame.display.flip()

    if game_over or game_won:
        pygame.time.wait(2000)
        running = False

pygame.quit()
