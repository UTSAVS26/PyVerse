import pygame
import random
import sys

# Initialize Pygame
pygame.init()

# Game constants
WIDTH, HEIGHT = 800, 600
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
CASTLE_COLOR = (139, 69, 19)  # Dark brown for the castle walls
TOWER_COLOR = (105, 105, 105)  # Gray for the towers
DOOR_COLOR = (139, 69, 19)  # Brown for the castle door

# Screen setup
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Castle Defense")

# Player stats
player_points = 10
castle_health = 100

# Trap and upgrade setup
traps = []
trap_cost = 5
upgrade_cost = 10

# Enemy setup
enemy_spawn_rate = 1000  # milliseconds
last_enemy_spawn_time = pygame.time.get_ticks()
enemies = []
enemy_speed = 2

# Fonts
font = pygame.font.Font(None, 36)

# Define castle_rect globally
castle_rect = pygame.Rect(WIDTH // 2 - 100, HEIGHT - 150, 200, 100)

# Game functions
def spawn_enemy():
    x = random.randint(0, WIDTH)
    y = -50
    enemy_rect = pygame.Rect(x, y, 40, 40)
    enemies.append(enemy_rect)

def draw_text(text, font, color, x, y):
    text_surface = font.render(text, True, color)
    screen.blit(text_surface, (x, y))

def check_traps():
    global player_points
    for trap in traps:
        for enemy in enemies:
            if trap.colliderect(enemy):
                enemies.remove(enemy)
                player_points += 5  # Earn points for defeating an enemy

def draw_castle():
    # Use the global castle_rect
    global castle_rect
    # Castle main wall
    pygame.draw.rect(screen, CASTLE_COLOR, castle_rect)

    # Castle door
    door_rect = pygame.Rect(WIDTH // 2 - 25, HEIGHT - 100, 50, 50)
    pygame.draw.rect(screen, DOOR_COLOR, door_rect)

    # Left tower
    left_tower = pygame.Rect(WIDTH // 2 - 130, HEIGHT - 180, 60, 130)
    pygame.draw.rect(screen, TOWER_COLOR, left_tower)
    pygame.draw.rect(screen, BLACK, (WIDTH // 2 - 130, HEIGHT - 180, 60, 30))  # Tower top

    # Right tower
    right_tower = pygame.Rect(WIDTH // 2 + 70, HEIGHT - 180, 60, 130)
    pygame.draw.rect(screen, TOWER_COLOR, right_tower)
    pygame.draw.rect(screen, BLACK, (WIDTH // 2 + 70, HEIGHT - 180, 60, 30))  # Tower top

    # Add windows to the castle
    pygame.draw.rect(screen, BLACK, (WIDTH // 2 - 75, HEIGHT - 120, 20, 20))
    pygame.draw.rect(screen, BLACK, (WIDTH // 2 + 55, HEIGHT - 120, 20, 20))

# Game loop
clock = pygame.time.Clock()
running = True

while running:
    screen.fill(WHITE)
    
    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # Place trap on mouse click if player has enough points
            if player_points >= trap_cost:
                x, y = pygame.mouse.get_pos()
                trap_rect = pygame.Rect(x, y, 20, 20)
                traps.append(trap_rect)
                player_points -= trap_cost
                print(f"Trap placed at ({x}, {y}). Remaining points: {player_points}")
            else:
                print("Not enough points to place a trap.")

    # Enemy spawning
    current_time = pygame.time.get_ticks()
    if current_time - last_enemy_spawn_time > enemy_spawn_rate:
        spawn_enemy()
        last_enemy_spawn_time = current_time

    # Update enemies
    for enemy in enemies:
        enemy.y += enemy_speed
        if enemy.colliderect(castle_rect):  # Check collision with castle_rect
            castle_health -= 10  # Reduce castle health if an enemy reaches it
            enemies.remove(enemy)

    # Check traps
    check_traps()

    # Draw game objects
    draw_castle()  # Draw the new castle
    for enemy in enemies:
        pygame.draw.rect(screen, RED, enemy)
    for trap in traps:
        pygame.draw.rect(screen, BLUE, trap)
    
    # Draw HUD
    draw_text(f"Points: {player_points}", font, BLACK, 10, 10)
    draw_text(f"Castle Health: {castle_health}", font, BLACK, 10, 50)
    
    # Check game over condition
    if castle_health <= 0:
        draw_text("Game Over", font, BLACK, WIDTH // 2 - 50, HEIGHT // 2)
        pygame.display.flip()
        pygame.time.delay(2000)
        running = False
    
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()
