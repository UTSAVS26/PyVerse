import pygame
import random

# initialize Pygame
pygame.init()

# set up the screen
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pong")

# colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# game variables
player_width, player_height = 10, 100
ball_size = 20
player_speed = 10
ball_speed = 5
player_score = 0
opponent_score = 0

# font
font = pygame.font.Font(None, 48)

# paddle positions
player_pos = [50, HEIGHT // 2 - player_height // 2]
opponent_pos = [WIDTH - 50 - player_width, HEIGHT // 2 - player_height // 2]

# ball position and velocity
ball_pos = [WIDTH // 2, HEIGHT // 2]
ball_vel = [ball_speed, ball_speed]

clock = pygame.time.Clock()


def draw_text(text, color, x, y):
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect(center=(x, y))
    screen.blit(text_surface, text_rect)


def draw_paddle(pos):
    pygame.draw.rect(screen, WHITE, (pos[0], pos[1], player_width, player_height))


def draw_ball(pos):
    pygame.draw.circle(screen, WHITE, pos, ball_size)


def move_paddle(pos, direction):
    pos[1] += direction * player_speed
    if pos[1] < 0:
        pos[1] = 0
    elif pos[1] > HEIGHT - player_height:
        pos[1] = HEIGHT - player_height


def move_ball(pos, vel):
    pos[0] += vel[0]
    pos[1] += vel[1]


def check_collision(ball_pos, ball_vel, player_pos, opponent_pos):
    if ball_pos[1] <= ball_size or ball_pos[1] >= HEIGHT - ball_size:
        ball_vel[1] = -ball_vel[1]

    if ball_pos[0] <= player_pos[0] + player_width and \
            player_pos[1] <= ball_pos[1] <= player_pos[1] + player_height and \
            ball_vel[0] < 0:
        ball_vel[0] = -ball_vel[0]

    if ball_pos[0] >= opponent_pos[0] - ball_size and \
            opponent_pos[1] <= ball_pos[1] <= opponent_pos[1] + player_height and \
            ball_vel[0] > 0:
        ball_vel[0] = -ball_vel[0]


def reset_ball():
    global ball_pos, ball_vel
    ball_pos = [WIDTH // 2, HEIGHT // 2]
    ball_vel[0] = random.choice([-ball_speed, ball_speed])
    ball_vel[1] = random.choice([-ball_speed, ball_speed])


def move_opponent(opponent_pos, ball_pos):
    if opponent_pos[1] + player_height // 2 < ball_pos[1]:
        opponent_pos[1] += player_speed
    elif opponent_pos[1] + player_height // 2 > ball_pos[1]:
        opponent_pos[1] -= player_speed


def main():
    global player_score, opponent_score

    running = True

    while running:
        screen.fill(BLACK)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            move_paddle(player_pos, -1)
        if keys[pygame.K_DOWN]:
            move_paddle(player_pos, 1)

        move_opponent(opponent_pos, ball_pos)

        move_ball(ball_pos, ball_vel)

        check_collision(ball_pos, ball_vel, player_pos, opponent_pos)

        if ball_pos[0] <= 0:
            opponent_score += 1
            reset_ball()
        elif ball_pos[0] >= WIDTH:
            player_score += 1
            reset_ball()

        draw_paddle(player_pos)
        draw_paddle(opponent_pos)
        draw_ball(ball_pos)

        draw_text(str(player_score), WHITE, WIDTH // 4, 50)
        draw_text(str(opponent_score), WHITE, WIDTH * 3 // 4, 50)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    quit()


if __name__ == "__main__":
    main()
