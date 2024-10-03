# game.py

from chessboard import Chessboard
from player import Player

class Game:
    def __init__(self):
        self.chessboard = Chessboard()
        self.players = [Player("white"), Player("black")]
        self.turn = 0  # 0 for white, 1 for black

    def switch_turn(self):
        self.turn = 1 - self.turn

    def play(self):
        while True:
            self.chessboard.display()
            current_player = self.players[self.turn]
            # Get player input for moves (not fully implemented)
            from_pos = (0, 0)  # Example input
            to_pos = (1, 0)    # Example input
            current_player.make_move(self.chessboard, from_pos, to_pos)
            self.switch_turn()
