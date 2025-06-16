import cv2
from utils.constants import HEIGHT,PADDLE_HEIGHT,PADDLE_WIDTH

class Paddle:
    def __init__(self, x, y,color):
        self.x = x
        self.y = y
        self.width = PADDLE_WIDTH
        self.height = PADDLE_HEIGHT
        self.color = color

    def draw(self, frame):
        cv2.rectangle(
            frame,
            (int(self.x - self.width // 2), int(self.y - self.height // 2)),
            (int(self.x + self.width // 2), int(self.y + self.height // 2)),
            self.color,
            -1,
        )

    def move(self, x, y):
        self.x = x
        self.y = y

        # Ensure the paddle stays within the frame boundaries
        if self.y - self.height//2 <= 0:
            self.y = self.height//2
        if self.y + self.height//2 >= HEIGHT:
            self.y = HEIGHT - self.height//2
        

    # def reset(self):
    #     self.x = self. original_x
    #     self.y = self.original_y