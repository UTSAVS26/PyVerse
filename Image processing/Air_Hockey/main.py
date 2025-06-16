import cv2
from utils.hand_detection import HandDetection
from utils.Ball import Ball
from utils.Paddle import Paddle
from utils.collision import handle_collision
from utils.constants import WIDTH, HEIGHT, LEFT_OFFSET, RIGHT_OFFSET
from utils.Score import Score

# Initialize video capture
vid = cv2.VideoCapture(0)

# Create an instance of HandDetection
hand_detection = HandDetection()
hand_detection.create_trackbars()

# Create an instance of Score
score = Score()


def main():
    left_paddle = Paddle(LEFT_OFFSET, HEIGHT // 2, (255, 0, 0))
    right_paddle = Paddle(RIGHT_OFFSET, HEIGHT // 2, (0, 255, 0))
    ball = Ball(WIDTH // 2, HEIGHT // 2)

    while vid.isOpened():
        ret, frame = vid.read()
        if not ret:
            break

        # Resize to defined size
        frame = cv2.resize(frame, (WIDTH, HEIGHT))

        # Flip the frame for
        frame = cv2.flip(frame, 1)

        # Get Centroids of hands/color
        centroids = hand_detection.get_centroid(frame)

        # Assign centroids to paddles
        if len(centroids) == 1:
            left_paddle.move(left_paddle.x, centroids[0][1])
        elif len(centroids) == 2:
            # Sort by x-coordinate
            centroids = sorted(centroids, key=lambda c: c[0])
            left_paddle.move(left_paddle.x, centroids[0][1])
            right_paddle.move(right_paddle.x, centroids[1][1])
        
        # Draw paddles
        left_paddle.draw(frame)
        right_paddle.draw(frame)

        # Update physics & collisions first
        handle_collision(ball, left_paddle, right_paddle, frame)

        # Now evaluate scoring / render HUD
        score.show(ball, frame)

        cv2.imshow("Hand Gesture Slider", frame)

        # Exit
        key = cv2.waitKey(10)
        if key == ord("q"):
            break

    # Release the video capture and close all OpenCV windows
    vid.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()