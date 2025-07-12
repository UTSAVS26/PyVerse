"""
Problem: Count Square Submatrices with All Ones
Difficulty: Medium

This problem gives you a matrix of 0's and 1's. You need to count the total number of square submatrices with all ones.

Approach:
For each cell, if it is 1, the size of the largest square ending at that cell is 1 plus the minimum of the squares ending to the left, top, and top-left. Sum up all the sizes for all cells to get the total number of squares.
"""

def countSquares(matrix):
    m, n = len(matrix), len(matrix[0])  # Get the dimensions of the matrix
    dp = [[0]*n for _ in range(m)]  # DP table to store largest square ending at (i, j)
    res = 0  # Result to store the total number of squares
    for i in range(m):
        for j in range(n):
            if matrix[i][j] == 1:
                if i == 0 or j == 0:
                    # First row or first column, only 1x1 square possible
                    dp[i][j] = 1
                else:
                    # Minimum of top, left, and top-left neighbors + 1
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
                res += dp[i][j]  # Add the number of squares ending at (i, j)
            # If cell is 0, dp[i][j] remains 0
    return res  # Return the total number of squares 