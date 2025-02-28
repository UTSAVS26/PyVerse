def printSolution(board, N):
    # Print the board with queens represented by "Q" and empty spaces by "."
    for i in range(N):
        for j in range(N):
            print("Q" if board[i][j] == 1 else ".", end=" ")
        print()  # Newline for the next row


def isSafe(board, row, col, N):
    # Check if it's safe to place a queen at board[row][col]

    # Check this row on the left side for any queens
    for i in range(col):
        if board[row][i] == 1:
            return False

    # Check upper diagonal on the left side for any queens
    for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
        if board[i][j] == 1:
            return False

    # Check lower diagonal on the left side for any queens
    for i, j in zip(range(row, N, 1), range(col, -1, -1)):
        if board[i][j] == 1:
            return False

    return True  # If all checks passed, it's safe


def solveNQUtil(board, col, N):
    # Base case: If all queens are placed successfully, return true
    if col >= N:
        return True

    # Consider this column and try placing a queen in all rows
    for i in range(N):
        if isSafe(board, i, col, N):  # Check if placing a queen is safe
            board[i][col] = 1  # Place the queen

            # Recur to place the rest of the queens
            if solveNQUtil(board, col + 1, N):
                return True  # If successful, return true

            # If placing the queen doesn't lead to a solution, backtrack
            board[i][col] = 0  # Remove the queen

    return False  # If no position is found, return false


def solveNQ(N):
    # Initialize the board as a 2D array with 0s
    board = [[0] * N for _ in range(N)]

    # Start solving the N Queens problem
    if not solveNQUtil(board, 0, N):
        print("Solution does not exist")  # If no solution is found
        return False

    printSolution(board, N)  # Print the solution
    return True


# Driver Code
if __name__ == '__main__':
    N = int(input("Enter the number of queens (N): "))  # Get the number of queens from the user
    solveNQ(N)  # Solve the N Queens problem
