# chessboard.py

class Chessboard:
    def __init__(self):
        self.board = self.initialize_board()

    def initialize_board(self):
        # Create a standard chessboard with pieces
        board = [[None] * 8 for _ in range(8)]
        # Add pawns and other pieces to the board
        for i in range(8):
            board[1][i] = Pawn("white")  # White pawns
            board[6][i] = Pawn("black")  # Black pawns
        # Add other pieces (not fully implemented)
        return board

    def display(self):
        for row in self.board:
            print(" | ".join([str(piece) if piece else " . " for piece in row]))
            print("-" * 32)
