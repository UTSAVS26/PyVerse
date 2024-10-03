# pieces.py

class Piece:
    def __init__(self, color):
        self.color = color

    def get_valid_moves(self, position):
        raise NotImplementedError("This method should be overridden by subclasses")


class Pawn(Piece):
    def get_valid_moves(self, position):
        direction = 1 if self.color == "white" else -1
        # Example movement logic for pawn (not fully implemented)
        return [(position[0] + direction, position[1])]  # Move forward


class Rook(Piece):
    def get_valid_moves(self, position):
        # Rook movement logic (simplified)
        moves = []
        for i in range(1, 8):
            moves.append((position[0] + i, position[1]))  # Horizontal moves
            moves.append((position[0] - i, position[1]))  # Horizontal moves
            moves.append((position[0], position[1] + i))  # Vertical moves
            moves.append((position[0], position[1] - i))  # Vertical moves
        return moves
