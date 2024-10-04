import pygame
import os
import random
import sys

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 576
SCREEN_HEIGHT = 720
FPS = 120
GRAVITY = 0.1
BIRD_JUMP_STRENGTH = -5
PIPE_SPEED = 5
PIPE_SPAWN_INTERVAL = 2000
PIPE_GAP = 150
BIRD_FLAP_INTERVAL = 200
SCORE_SOUND_INTERVAL = 100

# Colors
WHITE = (255, 255, 255)

# Set up the display
WIN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Aritro's Flappy Bird!!!")
pygame.display.set_icon(pygame.image.load("favicon.ico"))
clock = pygame.time.Clock()

# Fonts
FONT_SMALL = pygame.font.SysFont('comicsans', 20)
FONT_MEDIUM = pygame.font.SysFont('comicsans', 30)
FONT_LARGE = pygame.font.SysFont('comicsans', 60)

# Load assets
def load_image(name):
    return pygame.image.load(os.path.join("sprites", name)).convert_alpha()

def load_scaled_image(name, scale=2):
    img = load_image(name)
    return pygame.transform.scale2x(img) if scale == 2 else pygame.transform.scale(img, scale)

bg_surface = load_scaled_image("background-day.png", (SCREEN_WIDTH, SCREEN_HEIGHT))
floor_surface = load_scaled_image("base.png", (SCREEN_WIDTH, 100))
pipe_surface = load_scaled_image("pipe-green.png")

bird_frames = [load_scaled_image(f"bluebird-{flap}flap.png") for flap in ('down', 'mid', 'up')]

# Load sounds
flap_sound = pygame.mixer.Sound(os.path.join('audio', 'wing.wav'))
die_sound = pygame.mixer.Sound(os.path.join('audio', 'die.wav'))
hit_sound = pygame.mixer.Sound(os.path.join('audio', 'hit.wav'))
score_sound = pygame.mixer.Sound(os.path.join('audio', 'point.wav'))

# Game variables
bird_movement = 0
game_active = True
score = 0
high_score = 0
floor_x_pos = 0
bird_index = 0
bird_surface = bird_frames[bird_index]
bird_rect = bird_surface.get_rect(center=(100, SCREEN_HEIGHT // 2))
pipe_list = []
score_sound_countdown = SCORE_SOUND_INTERVAL

# Custom events
BIRDFLAP = pygame.USEREVENT + 1
SPAWNPIPE = pygame.USEREVENT + 2
pygame.time.set_timer(BIRDFLAP, BIRD_FLAP_INTERVAL)
pygame.time.set_timer(SPAWNPIPE, PIPE_SPAWN_INTERVAL)

def draw_floor():
    """Draw the moving floor."""
    WIN.blit(floor_surface, (floor_x_pos, SCREEN_HEIGHT - 100))
    WIN.blit(floor_surface, (floor_x_pos + SCREEN_WIDTH, SCREEN_HEIGHT - 100))

def create_pipe():
    """Create a new pair of pipes."""
    random_height = random.choice([200, 300, 400])
    bottom_pipe = pipe_surface.get_rect(midtop=(SCREEN_WIDTH + 100, random_height))
    top_pipe = pipe_surface.get_rect(midbottom=(SCREEN_WIDTH + 100, random_height - PIPE_GAP))
    return bottom_pipe, top_pipe

def move_pipes(pipes):
    """Move pipes to the left."""
    return [pipe.move(-PIPE_SPEED, 0) for pipe in pipes]

def draw_pipes(pipes):
    """Draw pipes on the screen."""
    for pipe in pipes:
        if pipe.bottom >= SCREEN_HEIGHT:
            WIN.blit(pipe_surface, pipe)
        else:
            flip_pipe = pygame.transform.flip(pipe_surface, False, True)
            WIN.blit(flip_pipe, pipe)

def check_collision(pipes):
    """Check for collisions between the bird and pipes or screen boundaries."""
    for pipe in pipes:
        if bird_rect.colliderect(pipe):
            hit_sound.play()
            return False
    
    if bird_rect.top <= -100 or bird_rect.bottom >= SCREEN_HEIGHT - 100:
        die_sound.play()
        return False
    
    return True

def rotate_bird(bird):
    """Rotate the bird based on its movement."""
    return pygame.transform.rotozoom(bird, -bird_movement * 3, 1)

def bird_animation():
    """Animate the bird's wings."""
    new_bird = bird_frames[bird_index]
    new_bird_rect = new_bird.get_rect(center=(100, bird_rect.centery))
    return new_bird, new_bird_rect

def score_display(game_state):
    """Display the score and high score."""
    if game_state == 'main_game':
        score_surface = FONT_MEDIUM.render(str(int(score)), True, WHITE)
        score_rect = score_surface.get_rect(center=(SCREEN_WIDTH // 2, 30))
        WIN.blit(score_surface, score_rect)
    elif game_state == 'game_over':
        score_surface = FONT_MEDIUM.render(f"Score: {int(score)}", True, WHITE)
        score_rect = score_surface.get_rect(center=(SCREEN_WIDTH // 2, 30))
        WIN.blit(score_surface, score_rect)

        game_over = FONT_LARGE.render("GAME OVER", True, WHITE)
        game_over_rect = game_over.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 50))
        WIN.blit(game_over, game_over_rect)

        restart_text = FONT_SMALL.render("PRESS 'ENTER' TO CONTINUE", True, WHITE)
        restart_rect = restart_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 20))
        WIN.blit(restart_text, restart_rect)

        high_score_surface = FONT_MEDIUM.render(f"High Score: {int(high_score)}", True, WHITE)
        high_score_rect = high_score_surface.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 30))
        WIN.blit(high_score_surface, high_score_rect)

def update_score(score):
    """Update and return the high score."""
    file_path = "score.txt"
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            f.write("0")
    
    with open(file_path, "r") as f:
        high_score = int(f.read())
    
    if score > high_score:
        high_score = score
        with open(file_path, "w") as f:
            f.write(str(int(high_score)))
    
    return high_score

# Main game loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE and game_active:
                bird_movement = BIRD_JUMP_STRENGTH
                flap_sound.play()
            if event.key == pygame.K_RETURN and not game_active:
                game_active = True
                pipe_list.clear()
                bird_rect.center = (100, SCREEN_HEIGHT // 2)
                bird_movement = 0
                score = 0
        if event.type == SPAWNPIPE:
            pipe_list.extend(create_pipe())
        if event.type == BIRDFLAP:
            bird_index = (bird_index + 1) % 3
            bird_surface, bird_rect = bird_animation()

    WIN.blit(bg_surface, (0, 0))

    if game_active:
        # Bird
        bird_movement += GRAVITY
        rotated_bird = rotate_bird(bird_surface)
        bird_rect.centery += bird_movement
        WIN.blit(rotated_bird, bird_rect)
        game_active = check_collision(pipe_list)

        # Pipes
        pipe_list = move_pipes(pipe_list)
        draw_pipes(pipe_list)
        
        # Score
        score += 0.01
        score_display('main_game')
        score_sound_countdown -= 1
        if score_sound_countdown <= 0:
            score_sound.play()
            score_sound_countdown = SCORE_SOUND_INTERVAL
    else:
        high_score = update_score(score)
        score_display('game_over')

    # Floor
    floor_x_pos -= 1
    draw_floor()
    if floor_x_pos <= -SCREEN_WIDTH:
        floor_x_pos = 0

    # Credits
    credits = FONT_SMALL.render("Game Developed by Aritro Saha", True, WHITE)
    credits_rect = credits.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 20))
    WIN.blit(credits, credits_rect)

    pygame.display.update()
    clock.tick(FPS)