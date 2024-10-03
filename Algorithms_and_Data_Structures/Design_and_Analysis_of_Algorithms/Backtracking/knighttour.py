n = 8

def is_safe(x, y, board):
    return 0 <= x < n and 0 <= y < n and board[x][y] == -1

def print_solution(board):
    for row in board:
        print(' '.join(map(str, row)))

def solve_knight_tour(n):
    board = [[-1 for _ in range(n)] for _ in range(n)]
    move_x = [2, 1, -1, -2, -2, -1, 1, 2]
    move_y = [1, 2, 2, 1, -1, -2, -2, -1]
    board[0][0] = 0
    pos = 1

    if not solve_knight_tour_util(n, board, 0, 0, move_x, move_y, pos):
        print("Solution does not exist")
    else:
        print_solution(board)

def solve_knight_tour_util(n, board, curr_x, curr_y, move_x, move_y, pos):
    if pos == n**2:
        return True

    for i in range(8):
        new_x = curr_x + move_x[i]
        new_y = curr_y + move_y[i]
        if is_safe(new_x, new_y, board):
            board[new_x][new_y] = pos
            if solve_knight_tour_util(n, board, new_x, new_y, move_x, move_y, pos + 1):
                return True
            board[new_x][new_y] = -1
    return False

def main():
    global n
    n = int(input("Enter the size of the chessboard (e.g., 8 for an 8x8 board): "))
    solve_knight_tour(n)

if __name__ == "__main__":
    main()
