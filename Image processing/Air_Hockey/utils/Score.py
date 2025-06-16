import cv2
from utils.constants import LEFT_OFFSET,RIGHT_OFFSET,WIDTH,MAX_SCORE


class Score:
    def __init__(self):
        self.player_a = 0
        self.player_b = 0
        self.winner = ""
        pass

    def isWinner(self):
        if(self.player_a>=MAX_SCORE):
            self.winner = "A"
            return True
        elif(self.player_b>=MAX_SCORE):
            self.winner = "B"
            return True
        else:
            return False
        
    def calculate_score(self, ball):
        if ball.x - ball.radius < LEFT_OFFSET:
            self.player_b += 1
        elif ball.x + ball.radius > RIGHT_OFFSET:
            self.player_a += 1
        pass

    def reset(self):
        self.player_a = 0
        self.player_b = 0
        self.winner = ""

    def show(self, ball, frame):
        self.calculate_score(ball)
        
        if(not self.isWinner()):
            cv2.putText(
                frame,
                f"Player A: {self.player_a}",
                (WIDTH//7, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Player B: { self.player_b}",
                (WIDTH*4//7, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
        else:
            cv2.putText(
                frame,
                f"Winner is Player: {self.winner}!!!",
                (WIDTH*2//7, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            cv2.imshow("Hand Gesture Slider", frame)
            cv2.waitKey(5000)
            self.reset()
            

    pass