
class Piece:
    
    def __init__(self ,color) -> None:
        self.color = color
        self.has_moved = False
        #print("Piece class constructor called")
    
    """ each class of piece will have its own symbol and this method will be overriden in every subclass"""
    def symbol(self) -> str:
        return '?'

    """ checks move is acceptable or not"""
    def is_valid_move(self ,start_position:tuple[int ,int],end_position:tuple[int ,int],board:list[list['Piece']]) -> bool:
        raise NotImplementedError("method should be overriden in base classes")
    
    def get_color(self):
        return self.color
    
class Pawn(Piece):
    
    def __init__(self ,color) -> None:
        super().__init__(color)
    
    def symbol(self) -> str:
        return 'P' if self.color == 'white' else 'p'

    def is_valid_move(self ,start_position:tuple[int ,int] ,end_position:tuple[int ,int] ,board:list[list['Piece']]) -> bool:
        """ pawn moves one step forward and two step forward is only available when it is the first move of that pawn
            if pawn is in (i,j) it can move to (i-1 ,j) for black (assumption black is at bottom) and (i+1 ,j) for white
            edge case if it is at the first row or at the last row :- it cannot move further
        """
        row_initial ,col_initial = start_position[0] ,start_position[1]
        row_final ,col_final = end_position[0] ,end_position[1]
        
        """ edge case - avoid pawn to going off the board """
        if not (0 <= row_final < 8 and 0 <= col_final < 8):
            return False

        """ Handling logic for white pawn"""
        if self.color == 'white':
            #One square forward 
            if col_initial == col_final and row_final == row_initial + 1 and board[row_final][col_final] is None:
                return True
            
            # Two squares forward on the first move
            elif col_initial == col_final and row_final == row_initial + 2 and not self.has_moved and board[row_initial + 1][col_initial] is None and board[row_final][col_final] is None:
                return True
            
            # Diagonal capture
            elif abs(col_initial - col_final) == 1 and row_final == row_initial + 1 and board[row_final][col_final] is not None and board[row_final][col_final].color != self.color:
                return True
            
            else:
                return False
        else:
            #One square forward
            if col_initial == col_final and row_final == row_initial - 1 and board[row_final][col_final] is None:
                return True
            
            # Two squares forward on the first move
            elif col_initial == col_final and row_final == row_initial - 2 and not self.has_moved and board[row_initial - 1][col_initial] is None and board[row_final][col_final] is None:
                return True
            #Diagonal capture
            elif abs(col_initial - col_final) == 1 and row_final == row_initial - 1 and board[row_final][col_final] is not None and board[row_final][col_final].color != self.color:
                return True
            else:
                return False
            
class Rook(Piece):
    
    def __init__(self ,color):
        super().__init__(color)
    
    def symbol(self) -> str:
        return 'R' if self.color == 'white' else 'r'

    def is_valid_move(self ,start_position:tuple[int ,int] ,end_position:tuple[int ,int] ,board:list[list['Piece']]) -> bool:
        """ rooks can move horizontally and vertically 
            means (i,j) - (i-k,j) or (i+k,j) for k = 1 to 7
            or (i,j) - (i,j-k) or (i,j+k) for k = 1 to 7 
        """
        
        row_initial ,col_initial = start_position[0] ,start_position[1]
        row_final ,col_final = end_position[0] ,end_position[1]
        
        # Ensure the move stays on the board
        if not (0 <= row_final < len(board) and 0 <= col_final < len(board[0])):
            return False
        
        # Rooks only move either horizontally or vertically, not diagonally
        if row_initial != row_final and col_initial != col_final:
            return False

        # Check for horizontal movement
        if row_initial == row_final:
            step = 1 if col_final > col_initial else -1
            for col in range(col_initial + step, col_final, step):
                if board[row_initial][col] is not None:  # Check for obstacles in the path
                    return False
            return True  # The move is valid if it reaches here

        # Check for vertical movement
        elif col_initial == col_final:
            step = 1 if row_final > row_initial else -1
            for row in range(row_initial + step, row_final, step):
                if board[row][col_initial] is not None:  # Check for obstacles in the path
                    return False
            return True  # The move is valid if it reaches here

        return False  # If none of the conditions matched, it's not a valid move

    
class Knight(Piece):
    
    def __init__(self ,color):
        super().__init__(color)
    
    def symbol(self) -> str:
        return 'KN' if self.color == 'white' else 'kn'

    def is_valid_move(self ,start_position:tuple[int ,int] ,end_position:tuple[int ,int] ,board:list[list['Piece']]) -> bool:
        """  """
        row_initial ,col_initial = start_position[0] ,start_position[1]
        row_final ,col_final = end_position[0] ,end_position[1]
        
        if not (0 <= row_final < len(board)) or (0 <= col_final < len(board[0])):
            return False
        
        """ check for forward movements """
        if(row_final<row_initial):
            if row_final == row_initial-1 and (col_final == col_initial-2 or col_final == col_initial+2) and board[row_final][col_final] is None:
                return True
                
            elif row_final == row_initial-2 and (col_final == col_initial-1 or col_final == col_initial+1) and board[row_final][col_final] is None:
                    return True
        else:
            """ backward movements """
            if row_final == row_initial+1 and (col_final == col_initial-2 or col_final == col_initial+2) and board[row_final][col_final] is None:
                return True
                
            elif row_final == row_initial+2 and (col_final == col_initial-1 or col_final == col_initial+1) and board[row_final][col_final] is None:
                return True
            
        return False

class Bishop(Piece):
    
    def __init__(self ,color):
        super().__init__(color)
    
    def symbol(self) -> str:
        return 'B' if self.color == 'white' else 'b'

    def is_valid_move(self ,start_position:tuple[int ,int] ,end_position:tuple[int ,int] ,board:list[list['Piece']]) -> bool:
        
        row_initial ,col_initial = start_position[0] ,start_position[1]
        row_final ,col_final = end_position[0] ,end_position[1]
        
        if not (0 <= row_final < len(board)) or (0 <= col_final < len(board[0])):
            return False
         # Check if the move is diagonal
        if abs(row_final - row_initial) != abs(col_final - col_initial):
            return False
        
        # Check if the path is clear
        row_step = 1 if row_final > row_initial else -1
        col_step = 1 if col_final > col_initial else -1
        
        current_row = row_initial + row_step
        current_col = col_initial + col_step
        
        while current_row != row_final and current_col != col_final:
            if board[current_row][current_col] is not None:
                return False  # Blocked by another piece
            current_row += row_step
            current_col += col_step
        
        return True  # Valid move

class Queen(Piece):
    
    def __init__(self, color):
        super().__init__(color)
    
    def symbol(self) -> str:
        return 'Q' if self.color == 'white' else 'q'

    def is_valid_move(self, start_position: tuple[int, int], end_position: tuple[int, int], board: list[list['Piece']]) -> bool:
        
        row_initial ,col_initial = start_position[0] ,start_position[1]
        row_final ,col_final = end_position[0] ,end_position[1]
        
        # Check if the end position is within the board boundaries
        if not (0 <= row_final < len(board)) or not (0 <= col_final < len(board[0])):
            return False
        
        # Check for movement in straight lines (row or column) or diagonally
        if row_initial == row_final or col_initial == col_final or abs(row_final - row_initial) == abs(col_final - col_initial):
            # Check for obstacles in the path
            step_row = (row_final - row_initial) // max(1, abs(row_final - row_initial)) if row_initial != row_final else 0
            step_col = (col_final - col_initial) // max(1, abs(col_final - col_initial)) if col_initial != col_final else 0
            
            current_row = row_initial + step_row
            current_col = col_initial + step_col
            
            while (current_row, current_col) != (row_final, col_final):
                if board[current_row][current_col] is not None:
                    return False  # There's an obstacle in the path
                current_row += step_row
                current_col += step_col
            
            return True
        
        return False


class King(Piece):
    
    def __init__(self, color):
        super().__init__(color)
    
    def symbol(self) -> str:
        return 'K' if self.color == 'white' else 'k'

    def is_valid_move(self, start_position: tuple[int, int], end_position: tuple[int, int], board: list[list['Piece']]) -> bool:
        
        row_initial ,col_initial = start_position[0] ,start_position[1]
        row_final ,col_final = end_position[0] ,end_position[1]
        
        # Check if the end position is within the board boundaries
        if not (0 <= row_final < len(board)) or not (0 <= col_final < len(board[0])):
            return False
        
        # Check if the move is one square away in any direction
        if max(abs(row_final - row_initial), abs(col_final - col_initial)) > 1:
            return False
        
        return True
