def printSolution(board, N):
    for i in range(N):
        for j in range(N):
            print("Q" if board[i][j] == 1 else ".", end=" ")
        print()


def isSafe(board, row, col, N):
    # Check this row on the left side
    for i in range(col):
        if board[row][i] == 1:
            return False

    # Check upper diagonal on the left side
    for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
        if board[i][j] == 1:
            return False

    # Check lower diagonal on the left side
    for i, j in zip(range(row, N, 1), range(col, -1, -1)):
        if board[i][j] == 1:
            return False

    return True


def solveNQUtil(board, col, N):
    # Base case: If all queens are placed, return true
    if col >= N:
        return True

    # Consider this column and try placing this queen in all rows
    for i in range(N):
        if isSafe(board, i, col, N):
            # Place this queen
            board[i][col] = 1

            # Recur to place rest of the queens
            if solveNQUtil(board, col + 1, N):
                return True

            # If placing queen doesn't lead to a solution, backtrack
            board[i][col] = 0

    return False


def solveNQ(N):
    # Initialize the board
    board = [[0] * N for _ in range(N)]

    # Start solving the N Queen problem
    if not solveNQUtil(board, 0, N):
        print("Solution does not exist")
        return False

    printSolution(board, N)
    return True


# Driver Code
if __name__ == '__main__':
    N = int(input("Enter the number of queens (N): "))
    solveNQ(N)
