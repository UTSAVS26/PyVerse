from utils.constants import HEIGHT, LEFT_OFFSET, RIGHT_OFFSET


def handle_collision(ball, left_paddle, right_paddle, frame):
    # Collision with side edges
    if ball.x - ball.radius < LEFT_OFFSET or ball.x + ball.radius > RIGHT_OFFSET:
        ball.reset()

    # Collision with top edge
    if ball.y - ball.radius <= 0:
        ball.y_vel = -ball.y_vel

    # Collision with bottom edge
    if ball.y + ball.radius >= HEIGHT:
        ball.y_vel = -ball.y_vel

    # Collision with the right slider
    if ball.x + ball.radius >= (right_paddle.x) - right_paddle.width // 2 and (
        ball.y + ball.radius >= right_paddle.y - right_paddle.height // 2 
        and ball.y - ball.radius <= right_paddle.y + right_paddle.height // 2
    ):
        ball.x_vel = -ball.x_vel

        ball.increase_speed()

    # Collision with left slider
    if ball.x - ball.radius <= left_paddle.x + left_paddle.width // 2 and (
        ball.y + ball.radius >= left_paddle.y - left_paddle.height // 2
        and ball.y - ball.radius <= left_paddle.y + left_paddle.height // 2
    ):
        ball.x_vel = -ball.x_vel

        ball.increase_speed()

    ball.move(frame)