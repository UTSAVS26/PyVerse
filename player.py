# player.py

class Player:
    def __init__(self, color):
        self.color = color

    def make_move(self, chessboard, from_pos, to_pos):
        piece = chessboard.board[from_pos[0]][from_pos[1]]
        if piece and piece.color == self.color:
            valid_moves = piece.get_valid_moves(from_pos)
            if to_pos in valid_moves:
                # Move the piece (simplified)
                chessboard.board[to_pos[0]][to_pos[1]] = piece
                chessboard.board[from_pos[0]][from_pos[1]] = None
