import pygame
import random
import pygame.mixer


# Colors for the game
RED = (255, 51, 51)                  #for the obstacle blocks
LIGHT_BLUE = (0, 204, 204)           #for the background
GREEN = (51, 255, 51)                #for the player block
BLACK = (0, 0, 0)                    #for the walls
DARK_BLUE = (15, 15, 200)            #for the text


# Game sounds and music
welcome_screen_music = 'audios/welcome_screen_music.wav'
in_game_music = 'audios/in_game_music.ogg'
level_up_sound = 'audios/level_up.ogg'
crash_sound = 'audios/crash.flac'
game_over_sound = 'audios/game_over.mp3'  
game_completed_sound = 'audios/game_completed.ogg'


# Welcome screen
def start_screen(screen):
    pygame.mixer.music.play(-1)
    screen.fill(LIGHT_BLUE)

    # Font
    title_font = pygame.font.SysFont('comicsansms', 60, bold=True)
    font = pygame.font.SysFont('comicsansms', 30)
    
    # Instructions 
    title_text = "HOW TO PLAY"
    instructions = [
        "1. Use arrow keys to move the player (Green block).",
        "2. Colliding with red blocks will end the game.",
        "3. Avoid touching the walls. It will reduce your score.",
        "4. Reach the goal to win and advance to the next level."
    ]
    
    # Calculate text position to be centered
    screen_rect = screen.get_rect()
    
    # Title positioning
    title_surface = title_font.render(title_text, True, DARK_BLUE)
    title_rect = title_surface.get_rect(center=(screen_rect.centerx, screen_rect.centery - 150))
    screen.blit(title_surface, title_rect)
    
    # Display instructions line by line and center align
    for i, line in enumerate(instructions):
        text_surface = font.render(line, True, DARK_BLUE)
        text_rect = text_surface.get_rect(center=(screen_rect.centerx, screen_rect.centery - 50 + i * 50))
        screen.blit(text_surface, text_rect)
    
    # Create and display the Start button
    button_color = GREEN
    button_rect = pygame.Rect(0, 0, 300, 70)
    button_rect.center = (screen_rect.centerx, screen_rect.centery + 200)
    
    pygame.draw.rect(screen, button_color, button_rect)
    pygame.draw.rect(screen, BLACK, button_rect, 3)  # Add border
    
    start_text = font.render("START", True, BLACK)
    start_text_rect = start_text.get_rect(center=button_rect.center)
    screen.blit(start_text, start_text_rect)
    
    pygame.display.flip()

    # Wait for user to click the Start button
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                if button_rect.collidepoint(mouse_pos):
                    waiting = False  # Exit the loop and start the game

        # Change the cursor to a pointer when hovering over the button
        mouse_pos = pygame.mouse.get_pos()
        if button_rect.collidepoint(mouse_pos):
            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND)  # Set cursor to pointer
        else:
            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)  # Set cursor to default arrow


# General utility functions:
def handle_player_movement(event, player):
    if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_LEFT:
            player.rect.x -= 3
        if event.key == pygame.K_RIGHT:
            player.rect.x += 3
        if event.key == pygame.K_UP:
            player.rect.y -= 3
        if event.key == pygame.K_DOWN:
            player.rect.y += 3

def display_score(screen, score, wall_touch_count, level):
    font = pygame.font.SysFont('comicsansms', 30)
    score_text = font.render(f"Score: {score - wall_touch_count}", True, DARK_BLUE)
    level_text = font.render(f"Level: {level}", True, DARK_BLUE)
    screen.blit(score_text, [10, 10])
    screen.blit(level_text, [10, 50])

def move_blocks(block, index, l1, l2):
    block.rect.x += l1[index]
    block.rect.y += l2[index]
    if block.rect.x >= 840 or block.rect.x < 10:
        l1[index] *= -1
    if block.rect.y >= 540 or block.rect.y < 10:
        l2[index] *= -1


# Block class definition
class Block(pygame.sprite.Sprite):
    
    def __init__(self, color, width, height):
        
        super().__init__()
        
        self.image = pygame.Surface([width, height])
        self.image.fill(color)
 
        self.rect = self.image.get_rect()


# Main game loop and logic
pygame.init()
pygame.mixer.init()
size = (900,600)
screen = pygame.display.set_mode((size))
pygame.display.set_caption("My Game")

pygame.mixer.music.load(welcome_screen_music)
start_screen(screen)
pygame.mixer.music.stop()
pygame.mixer.music.load(in_game_music)
pygame.mixer.music.play(-1)

# Main game loop and logic
size = (900,600)
screen = pygame.display.set_mode((size))
pygame.display.set_caption("My Game")


# Game variables
block_list = pygame.sprite.Group()                       #for the obstacle blocks
all_sprites_list = pygame.sprite.Group()                 #for the obstacle blocks and the player block
l1 = []
l2 = []


# Setup obstacle blocks
for i in range(1):
    block = Block(RED, 50, 50)
    block.rect.x = random.randrange(200,830)
    block.rect.y = random.randrange(200,530)
    block_list.add(block)
    all_sprites_list.add(block)

for k in range(15):
    xchange = random.randrange(0, 5)
    l1.append(xchange)
    ychange = random.randrange(0, 5)
    l2.append(ychange)

# Setup player
player = Block(GREEN, 20, 20)
player.rect.left = 11
player.rect.top = 11
all_sprites_list.add(player)

# Game loop
game_loop = True
clock = pygame.time.Clock()
wall_touch_count = 0 
score = 100
level = 1

while game_loop:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game_loop = False


    # Create the game world (bg and walls)
    screen.fill(LIGHT_BLUE)
    pygame.draw.line(screen,BLACK,[0,0],[0,600],20)
    pygame.draw.line(screen,BLACK,[0,0],[900,0],20)
    pygame.draw.line(screen,BLACK,[0,600],[900,600],20)
    pygame.draw.line(screen,BLACK,[900,0],[900,500],20)
    display_score(screen, score, wall_touch_count, level)


    # Collision detection
    blocks_hit_list = pygame.sprite.spritecollide(player, block_list, True)
    all_sprites_list.draw(screen)
    pygame.display.flip()


    # Player movement
    handle_player_movement(event, player)


    # Obstacle block movement
    i = 0 
    for block in block_list:
        move_blocks(block, i, l1, l2)
        i += 1


    if player.rect.left < 11:
        wall_touch_count += 1
        player.rect.left = 11
    if player.rect.right > 890 and player.rect.top <= 500:
        wall_touch_count += 1
        player.rect.right = 890
    if player.rect.top < 11:
        wall_touch_count += 1
        player.rect.top = 11
    if player.rect.bottom > 590:
        wall_touch_count += 1
        player.rect.bottom = 590
    if player.rect.left >= 900:
        score = score+1000
        level = level+1

        if level < 7:
            pygame.mixer.music.stop()
            pygame.mixer.Sound(level_up_sound).play()
            font = pygame.font.SysFont('comicsansms', 85, True, False)
            text = font.render("NEXT LEVEL", True, DARK_BLUE)
            screen.blit(text, [180, 200])
            pygame.display.flip()
            pygame.time.wait(3000)
            pygame.mixer.music.load(in_game_music)
            pygame.mixer.music.play(-1)
            

        if level == 7:                                                           #current game has 6 levels, can be increased
            pygame.mixer.music.stop()
            pygame.mixer.Sound(game_completed_sound).play()
            font = pygame.font.SysFont('comicsansms', 45, True, False)
            text = font.render("YOU COMPLETED THE GAME", True, DARK_BLUE)
            screen.blit(text, [150, 200])
            pygame.display.flip()
            pygame.time.wait(3000)

            # Clear the screen
            screen.fill(LIGHT_BLUE)

            pygame.mixer.Sound(game_over_sound).play()
            game_over_font = pygame.font.SysFont('comicsansms', 70, True, False)
            game_over_text = game_over_font.render("GAME OVER", True, DARK_BLUE)
            final_score_text = game_over_font.render(f"FINAL SCORE: {score - wall_touch_count}", True, DARK_BLUE)
            screen.blit(game_over_text, [230, 200])
            screen.blit(final_score_text, [100, 300])
            pygame.display.flip()
            pygame.time.wait(3000)

            FINAL_SCORE=(score-wall_touch_count)
            print('YOUR FINAL SCORE IS',FINAL_SCORE)
            game_loop = False

            
        for j in range(level):
            block = Block(RED, 50, 50)
            block.rect.x = random.randrange(100,830)
            block.rect.y = random.randrange(100,530)
            block_list.add(block)
            all_sprites_list.add(block)
            player.rect.left=11
            player.rect.top=11

    
    for block in blocks_hit_list:
        pygame.mixer.music.stop()
        pygame.mixer.Sound(crash_sound).play()
        font = pygame.font.SysFont('comicsansms', 85, True, False)
        text = font.render("YOU CRASHED", True, DARK_BLUE)
        screen.blit(text, [150, 200])
        pygame.display.flip()
        pygame.time.wait(3000)

        # Clear the screen
        screen.fill(LIGHT_BLUE)
        
        pygame.mixer.Sound(game_over_sound).play()
        game_over_font = pygame.font.SysFont('comicsansms', 70, True, False)
        game_over_text = game_over_font.render("GAME OVER", True, DARK_BLUE)
        final_score_text = game_over_font.render(f"FINAL SCORE: {score - wall_touch_count}", True, DARK_BLUE)
        screen.blit(game_over_text, [230, 200])
        screen.blit(final_score_text, [100, 300])
        pygame.display.flip()
        pygame.time.wait(3000)
        
        
        FINAL_SCORE=(score-wall_touch_count)
        print()
        print('GAME OVER\nYOUR FINAL SCORE IS',FINAL_SCORE)
        print()
        game_loop = False
        
    
    clock.tick(60)


pygame.quit()


