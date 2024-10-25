import pygame
import random
import sys

pygame.init()

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

CAR_WIDTH = 50
CAR_HEIGHT = 50
ZOMBIE_WIDTH = 30
ZOMBIE_HEIGHT = 30
POWER_UP_WIDTH = 20
POWER_UP_HEIGHT = 20

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Traffic vs Zombie')

# Load images
try:
    car_image = pygame.image.load('/workspaces/PyVerse/traffic_vs_zombie_game/images/car.jpeg').convert_alpha()
    zombie_image = pygame.image.load('/workspaces/PyVerse/traffic_vs_zombie_game/images/zombie.webp').convert_alpha()
    power_up_image = pygame.image.load('/workspaces/PyVerse/traffic_vs_zombie_game/images/powerup.png').convert_alpha()
except pygame.error as e:
    print(f"Error loading images: {e}")
    pygame.quit()
    sys.exit()

crash_sound = pygame.mixer.Sound('/workspaces/PyVerse/traffic_vs_zombie_game/sound/crash.wav.mp3')
power_up_sound = pygame.mixer.Sound('/workspaces/PyVerse/traffic_vs_zombie_game/sound/power up.wav.mp3')
pygame.mixer.music.load('/workspaces/PyVerse/traffic_vs_zombie_game/sound/background.wav.mp3')
pygame.mixer.music.play(-1)  

# Scale images to match object dimensions 
car_image = pygame.transform.scale(car_image, (CAR_WIDTH, CAR_HEIGHT))
zombie_image = pygame.transform.scale(zombie_image, (ZOMBIE_WIDTH, ZOMBIE_HEIGHT))
power_up_image = pygame.transform.scale(power_up_image, (POWER_UP_WIDTH, POWER_UP_HEIGHT))

# Car class
class Car:
    def _init_(self):
        self.x = SCREEN_WIDTH // 2 - CAR_WIDTH // 2  # Centered horizontally
        self.y = SCREEN_HEIGHT - CAR_HEIGHT - 10     # Positioned near the bottom
        self.speed = 5
        self.image = car_image
        self.rect = self.image.get_rect(topleft=(self.x, self.y))

    def move_left(self):
        if self.rect.left > 0:
            self.rect.x -= self.speed

    def move_right(self):
        if self.rect.right < SCREEN_WIDTH:
            self.rect.x += self.speed

    def draw(self):
        screen.blit(self.image, self.rect)

# Zombie class
class Zombie:
    def _init_(self):
        self.reset()

    def reset(self):
        # Ensure zombies spawn away from the car's initial position
        self.x = random.randint(0, SCREEN_WIDTH - ZOMBIE_WIDTH)
        self.y = -ZOMBIE_HEIGHT
        self.speed = random.randint(2, 6)
        self.image = zombie_image
        self.rect = self.image.get_rect(topleft=(self.x, self.y))

    def move_down(self):
        self.rect.y += self.speed

    def draw(self):
        screen.blit(self.image, self.rect)

# PowerUp class
class PowerUp:
    def _init_(self):
        self.reset()

    def reset(self):
        # Ensure power-ups spawn away from the car's initial position
        self.x = random.randint(0, SCREEN_WIDTH - POWER_UP_WIDTH)
        self.y = -POWER_UP_HEIGHT
        self.speed = 3
        self.image = power_up_image
        self.rect = self.image.get_rect(topleft=(self.x, self.y))

    def move_down(self):
        self.rect.y += self.speed

    def draw(self):
        screen.blit(self.image, self.rect)

# Function to show game over screen
def show_game_over_screen():
    font = pygame.font.Font(None, 74)
    text = font.render('GAME OVER', True, RED)
    screen.blit(text, (SCREEN_WIDTH // 2 - text.get_width() // 2, SCREEN_HEIGHT // 2 - text.get_height() // 2))
    pygame.display.flip()

    pygame.time.wait(2000)  # Wait for 2 seconds to display the game over message

    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:  # Press 'R' to restart the game
                    waiting = False
                if event.key == pygame.K_q:  # Press 'Q' to quit the game
                    pygame.quit()
                    sys.exit()

# Function to reset the game
def reset_game():
    global car, zombies, power_ups, score, level, game_over
    car = Car()
    zombies = [Zombie() for _ in range(5)]
    power_ups = [PowerUp() for _ in range(2)]
    score = 0
    level = 1
    game_over = False

# Initialize game objects
reset_game()

# Game loop
while True:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    if not game_over:
        # Move the car
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            car.move_left()
        if keys[pygame.K_RIGHT]:
            car.move_right()

        # Move the zombies
        for zombie in zombies:
            zombie.move_down()
            if zombie.rect.top > SCREEN_HEIGHT:
                zombie.reset()

        # Move the power-ups
        for power_up in power_ups:
            power_up.move_down()
            if power_up.rect.top > SCREEN_HEIGHT:
                power_up.reset()

        # Check for collisions with zombies
        collision_detected = False
        for zombie in zombies:
            if car.rect.colliderect(zombie.rect):
                print("Collision detected with zombie!")
                collision_detected = True
                break

        if collision_detected:
            game_over = True
            show_game_over_screen()
            reset_game()
            continue  # Skip drawing in this frame

        # Check for power-up collisions
        for power_up in power_ups:
            if car.rect.colliderect(power_up.rect):
                power_up.reset()  # Reset the power-up upon collection
                score += 10
                print(f"Power-up collected! Score: {score}")

        # Draw everything
        screen.fill(WHITE)
        car.draw()
        for zombie in zombies:
            zombie.draw()
        for power_up in power_ups:
            power_up.draw()

        # Display score and level
        font = pygame.font.Font(None, 36)
        text = font.render(f'Score: {score} | Level: {level}', True, BLACK)
        screen.blit(text, (10, 10))

        pygame.display.flip()

    else:
        show_game_over_screen()
        reset_game()

    pygame.time.delay(30)

pygame.quit()