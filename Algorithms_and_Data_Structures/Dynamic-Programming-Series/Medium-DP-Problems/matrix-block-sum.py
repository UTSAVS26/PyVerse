"""
Problem: Matrix Block Sum
Difficulty: Medium

This problem gives you a matrix and an integer K. For each cell, you need to calculate the sum of all elements within K distance from that cell.

Approach:
Build a prefix sum matrix so you can efficiently calculate the sum of any submatrix. For each cell, use the prefix sum to get the sum of the block in O(1) time.
"""

def matrixBlockSum(mat, K):
    m, n = len(mat), len(mat[0])  # Get the dimensions of the matrix
    # Build prefix sum matrix
    prefix = [[0]*(n+1) for _ in range(m+1)]  # Extra row and column for easier calculations
    for i in range(m):
        for j in range(n):
            # Each cell is sum of itself, left, top, and subtract top-left (to avoid double counting)
            prefix[i+1][j+1] = mat[i][j] + prefix[i][j+1] + prefix[i+1][j] - prefix[i][j]
    ans = [[0]*n for _ in range(m)]  # Result matrix
    for i in range(m):
        for j in range(n):
            # Calculate the boundaries of the block
            r1, c1 = max(0, i-K), max(0, j-K)
            r2, c2 = min(m, i+K+1), min(n, j+K+1)
            # Use prefix sum to get the sum of the block
            ans[i][j] = prefix[r2][c2] - prefix[r1][c2] - prefix[r2][c1] + prefix[r1][c1]
    return ans  # Return the result matrix 