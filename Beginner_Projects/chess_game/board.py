from piece import Piece ,King ,Knight ,Queen ,Rook ,Bishop ,Pawn

class Board:
    
    def __init__(self):
        #print("Board class constructor called")
        self.board = self.create_board()
        self.setup_pieces()
    
    def create_board(self):
        """ have to create a 8x8 matrix for the representation of chess_board """
        chess_board = []
        for _ in range(8):
            chess_board.append([None] * 8)        
        
        return chess_board

    def setup_pieces(self):
        """
            our board should look like :-
            
            R KN B Q KI B KN R
            P P  P P P  P P  P
            . .  . . .  . .  .
            . .  . . .  . .  .
            . .  . . .  . .  .
            . .  . . .  . .  .
            P P  P P P  P P  P
            R KN B Q KI B KN R
            
            Where   R  = Rook
                    KN = Knight
                    B  = Bishop
                    Q  = Queen
                    KI = King
                    P  = Pawn
        """
        
        """ set up pawn """
        for col in range(8):
            self.board[1][col] = Pawn('white')
            self.board[6][col] = Pawn('black')
        
        piece_order = [Rook ,Knight ,Bishop ,Queen ,King ,Bishop ,Knight ,Rook]
        
        """ setting up black piece """
        for col ,piece in enumerate(piece_order):
            self.board[0][col] = piece('white')
        
        """ setting up white piece """
        for col ,piece in enumerate(piece_order):
            self.board[7][col] = piece('black')
    
    def move_pieces(self ,start_position:tuple[int ,int] ,end_position:tuple[int ,int]) -> bool:
        
        row_initial ,col_initial = start_position[0],start_position[1]
        row_final ,col_final = end_position[0],end_position[1]
        
        piece = self.board[row_initial][col_initial]
        
        if piece is None:
            print("No piece at the given position")
            return False

        if piece.is_valid_move(start_position ,end_position ,self.board):
                
            target_piece = self.board[row_final][col_final]
            
            if target_piece is not None:
                if target_piece.color != piece.color:
                    print(f"Capturing {target_piece.symbol()} at {end_position}")
                else:
                    print("Invalid move: Cannot capture your own piece")
                    return False
            
            # Move the piece to the new location
            self.board[row_final][col_final] = piece
            self.board[row_initial][col_initial] = None
            print(f"Moved {piece.symbol()} to {end_position}")
            return True
        
        else:
            print("Invalid move")
            return False
    
    def print_board(self):
        """text-based representation of the board."""
        #column label
        print("  ", end='')
        for i in range(8):
            print(i, end=' ')
        print()  # newline after column labels
        # printing the board with row numbers
        for row_idx, row in enumerate(self.board):
            print(row_idx, end=' ')  # print row numbers (0 ,1, 2, 3, etc.)
            print(' '.join([piece.symbol() if piece else '.' for piece in row]))


obj = Board()
obj.print_board()
