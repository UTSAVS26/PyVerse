from piece import Piece ,Pawn ,Rook ,Bishop ,Queen ,King ,Knight
from board import Board

class Game:
    
    def __init__(self ,color):
        self.board = Board()
        self.current_color = color
        self.is_over = False
    
    def switch_turns(self):
        
        self.current_color = 'black' if self.current_color == "white" else "white"
    
    def is_checkmate(self ,color:str) -> bool:
        pass
    
    def is_in_check(self ,color:str) -> bool:
        
        king_position = None 
        """ finding king's position """
        for i in range(8):
            for j in range(8):
                piece = self.board.board[i][j]
                if piece is not None and isinstance(piece,King) and piece.color == color:
                    king_position = (i,j)
                    break
                
        opponent_color = 'black' if color == 'white' else 'white'
        for i in range(8):
            for j in range(8):
                piece = self.board.board[i][j]
                if piece is not None and piece.color == opponent_color:
                    if piece.is_valid_move((i,j),king_position,self.board.board):
                        return True # king is in check
        
        return False
    
    def play(self):
    
        while not self.is_over:
            self.board.print_board()
            print(f"{self.current_color.capitalize()} is playing")
        
            start_position = self.get_user_input("Enter the start position (row ,col)")
            if self.board.board[start_position[0]][start_position[1]] is not None:
                piece = self.board.board[start_position[0]][start_position[1]]
                print(f"{piece.color} {piece.__class__.__name__} selected")
            end_position = self.get_user_input("Enter the end position (row ,col)")
            
            if self.board.move_pieces(start_position ,end_position):
                if self.check_game_(self.current_color):
                    self.is_over = True
                self.switch_turns()
            
            else:
                print("Invalid move.. Please Take Another move")
    
    def get_user_input(self, prompt: str) -> tuple[int, int]:
        """take the input from user and convert to a tuple of integers."""
        while True:
            try:
                user_input = input(prompt)
                row, col = map(int, user_input.split(","))
                if 0 <= row < 8 and 0 <= col < 8:
                    return (row, col)
                else:
                    print("Invalid input.. Enter values between 0 and 7.")
            except ValueError:
                print("Invalid input format.. Use row,col format.")
    
    def check_game_(self ,color:str) -> bool:
        """ """
        if self.is_checkmate(color):
            pass
        if self.is_in_check(color):
            print(f"{color} King is in check")
        return False
        

if __name__ == "__main__":
    object = Game(color=input('Select color : black or white  '))
    object.play()
