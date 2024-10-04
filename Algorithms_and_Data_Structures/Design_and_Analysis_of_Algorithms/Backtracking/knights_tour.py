# Size of the chessboard
n = 8

def is_safe(x, y, board):
    # Check if the knight's move is within bounds and the position is not visited
    return 0 <= x < n and 0 <= y < n and board[x][y] == -1

def print_solution(board):
    # Print the chessboard with the knight's tour path
    for row in board:
        print(' '.join(map(str, row)))  # Print each row of the board

def solve_knight_tour(n):
    # Create a chessboard initialized with -1 (indicating unvisited)
    board = [[-1 for _ in range(n)] for _ in range(n)]
    
    # Possible moves for a knight (x, y offsets)
    move_x = [2, 1, -1, -2, -2, -1, 1, 2]
    move_y = [1, 2, 2, 1, -1, -2, -2, -1]
    
    board[0][0] = 0  # Starting position of the knight
    pos = 1  # Starting position index for the knight's tour

    # Start solving the knight's tour problem
    if not solve_knight_tour_util(n, board, 0, 0, move_x, move_y, pos):
        print("Solution does not exist")  # If no solution is found
    else:
        print_solution(board)  # Print the found solution

def solve_knight_tour_util(n, board, curr_x, curr_y, move_x, move_y, pos):
    # Base case: If all squares are visited
    if pos == n**2:
        return True

    # Try all possible knight moves
    for i in range(8):
        new_x = curr_x + move_x[i]  # New x coordinate after the move
        new_y = curr_y + move_y[i]  # New y coordinate after the move
        
        # Check if the new position is safe to move
        if is_safe(new_x, new_y, board):
            board[new_x][new_y] = pos  # Mark the new position with the move count
            
            # Recur to continue the tour
            if solve_knight_tour_util(n, board, new_x, new_y, move_x, move_y, pos + 1):
                return True  # If successful, return True
            
            board[new_x][new_y] = -1  # Backtrack: unmark the position
            
    return False  # No valid moves found, return False

def main():
    global n  # Declare n as global to modify it
    n = int(input("Enter the size of the chessboard (e.g., 8 for an 8x8 board): "))  # Get chessboard size from user
    solve_knight_tour(n)  # Start the knight's tour solution process

if __name__ == "__main__":
    main()  # Run the main function
