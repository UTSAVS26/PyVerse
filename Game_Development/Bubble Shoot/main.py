import pygame
import random
import math

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 600
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
BUBBLE_RADIUS = 20
BULLET_RADIUS = 5
FPS = 60

# Create the game window
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Bubble Hit Game")

# Define font
font = pygame.font.Font(None, 36)

# Clock to control frame rate
clock = pygame.time.Clock()

# Player bubble class
class PlayerBubble:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.speed = 5

    def draw(self):
        pygame.draw.circle(screen, BLUE, (self.x, self.y), BUBBLE_RADIUS)

    def move(self, direction):
        if direction == "LEFT" and self.x - BUBBLE_RADIUS > 0:
            self.x -= self.speed
        elif direction == "RIGHT" and self.x + BUBBLE_RADIUS < WIDTH:
            self.x += self.speed

# Bullet class
class Bullet:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.speed = 7

    def move(self):
        self.y -= self.speed

    def draw(self):
        pygame.draw.circle(screen, RED, (self.x, self.y), BULLET_RADIUS)

# Enemy bubble class
class EnemyBubble:
    def __init__(self):
        self.x = random.randint(BUBBLE_RADIUS, WIDTH - BUBBLE_RADIUS)
        self.y = random.randint(-100, -BUBBLE_RADIUS)
        self.speed = random.randint(2, 4)
    
    def move(self):
        self.y += self.speed

    def draw(self):
        pygame.draw.circle(screen, RED, (self.x, self.y), BUBBLE_RADIUS)

    def is_off_screen(self):
        return self.y - BUBBLE_RADIUS > HEIGHT

# Check for collision between bullets and enemies
def check_collision(bullet, enemy):
    distance = math.sqrt((bullet.x - enemy.x)**2 + (bullet.y - enemy.y)**2)
    return distance < BUBBLE_RADIUS + BULLET_RADIUS

# Game loop variables
player = PlayerBubble(WIDTH // 2, HEIGHT - 50)
bullets = []
enemies = []
score = 0
running = True

# Game loop
while running:
    # Fill background
    screen.fill(WHITE)

    # Display score
    score_text = font.render(f"Score: {score}", True, BLACK)
    screen.blit(score_text, (10, 10))

    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        # Fire bullet when space is pressed
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            bullets.append(Bullet(player.x, player.y - BUBBLE_RADIUS))

    # Handle player movement
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        player.move("LEFT")
    if keys[pygame.K_RIGHT]:
        player.move("RIGHT")

    # Move and draw bullets
    for bullet in bullets[:]:
        bullet.move()
        if bullet.y < 0:
            bullets.remove(bullet)
        else:
            bullet.draw()

    # Move and draw enemies
    if random.randint(1, 30) == 1:  # Random enemy spawn
        enemies.append(EnemyBubble())

    for enemy in enemies[:]:
        enemy.move()
        if enemy.is_off_screen():
            enemies.remove(enemy)
        else:
            enemy.draw()

        # Check for collisions
        for bullet in bullets[:]:
            if check_collision(bullet, enemy):
                bullets.remove(bullet)
                enemies.remove(enemy)
                score += 1
                break

    # Draw player
    player.draw()

    # Update display
    pygame.display.update()

    # Control frame rate
    clock.tick(FPS)

# Quit Pygame
pygame.quit()
