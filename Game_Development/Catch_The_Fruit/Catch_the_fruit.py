import pygame
import random


pygame.init()

SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Catch the Falling Fruits")

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)

clock = pygame.time.Clock()

BASKET_WIDTH, BASKET_HEIGHT = 100, 20
basket_x = SCREEN_WIDTH // 2 - BASKET_WIDTH // 2
basket_y = SCREEN_HEIGHT - BASKET_HEIGHT - 10
basket_speed = 10

FRUIT_SIZE = 30
fruit_speed = 5
fruits = []

score = 0
font = pygame.font.SysFont(None, 36)


def draw_basket(x, y):
    pygame.draw.rect(screen, GREEN, [x, y, BASKET_WIDTH, BASKET_HEIGHT])

def draw_fruit(fruit):
    pygame.draw.circle(screen, YELLOW, (fruit['x'], fruit['y']), FRUIT_SIZE // 2)

def display_score(score):
    score_text = font.render(f'Score: {score}', True, WHITE)
    screen.blit(score_text, (10, 10))

def generate_fruit():
    x = random.randint(0, SCREEN_WIDTH - FRUIT_SIZE)
    return {'x': x, 'y': -FRUIT_SIZE}

running = True
while running:
    screen.fill(BLACK)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT] and basket_x - basket_speed > 0:
        basket_x -= basket_speed
    if keys[pygame.K_RIGHT] and basket_x + BASKET_WIDTH + basket_speed < SCREEN_WIDTH:
        basket_x += basket_speed

    if random.randint(1, 30) == 1:
        fruits.append(generate_fruit())

    for fruit in fruits[:]:
        fruit['y'] += fruit_speed
        if fruit['y'] > SCREEN_HEIGHT:
            fruits.remove(fruit)
        elif (basket_y < fruit['y'] + FRUIT_SIZE and
              basket_x < fruit['x'] < basket_x + BASKET_WIDTH):
            fruits.remove(fruit)
            score += 10

    draw_basket(basket_x, basket_y)
    for fruit in fruits:
        draw_fruit(fruit)

    display_score(score)


    pygame.display.flip()
    clock.tick(30)

pygame.quit()
