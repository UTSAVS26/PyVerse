import cv2
import random
from utils.constants import SPEED_INCREMENT,BALL_RADIUS,BALL_VEL,BALL_COLOR

class Ball:
        def __init__(self, x, y):
            self.x = self.original_x = x
            self.y = self.original_y = y
            self.radius = BALL_RADIUS
            self.vel = BALL_VEL
            self.color = BALL_COLOR
            self.x_vel = BALL_VEL
            self.y_vel = -1*BALL_VEL

        def draw(self, frame):
            cv2.circle(frame, (int(self.x), int(self.y)),
            self.radius,self.color, -1)

        def move(self,frame):
            self.x += self.x_vel
            self.y += self.y_vel
            self.draw(frame)

        def increase_speed(self):
            self.x_vel *= SPEED_INCREMENT
            self.y_vel *= SPEED_INCREMENT

        def reset(self):
            self.x = self.original_x
            self.y = self.original_y
            choice = random.random()
            x_dir = 1 if choice < 0.5 else -1
            choice = random.random()
            y_dir = 1 if choice < 0.5 else -1
            self.x_vel = x_dir * self.vel 
            self.y_vel = y_dir * self.vel
