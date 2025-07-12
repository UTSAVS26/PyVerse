"""
Problem: Largest Plus Sign
Difficulty: Medium

This problem gives you an n x n grid with some cells as mines (0) and the rest as 1. You need to find the largest plus sign (center and 4 arms: up, down, left, right, all 1's). The order of the plus sign depends on the length of its arms.

Approach:
For each cell, count the number of continuous 1's in all 4 directions (up, down, left, right). The minimum of these counts is the order of the plus sign at that cell. The answer is the maximum order among all cells.
"""

def orderOfLargestPlusSign(n, mines):
    # Put all mines in a set for fast lookup
    banned = {tuple(mine) for mine in mines}  # O(mines) time

    # Create a DP matrix, initialize all cells to 0
    dp = [[0]*n for _ in range(n)]  # n x n matrix, all cells start at 0

    # For each row, count continuous 1's from left to right and right to left
    for r in range(n):
        count = 0
        for c in range(n):
            # If this cell is a mine, reset count to 0, else increment count
            count = 0 if (r, c) in banned else count + 1
            dp[r][c] = count  # store left-to-right count
        count = 0
        for c in range(n-1, -1, -1):
            # Same logic for right to left
            count = 0 if (r, c) in banned else count + 1
            dp[r][c] = min(dp[r][c], count)  # take the minimum (left/right)

    # For each column, count continuous 1's from top to bottom and bottom to top
    for c in range(n):
        count = 0
        for r in range(n):
            count = 0 if (r, c) in banned else count + 1
            dp[r][c] = min(dp[r][c], count)  # top to bottom
        count = 0
        for r in range(n-1, -1, -1):
            count = 0 if (r, c) in banned else count + 1
            dp[r][c] = min(dp[r][c], count)  # bottom to top

    # Find the largest plus sign order in the grid
    ans = 0
    for row in dp:
        ans = max(ans, max(row))  # get the max value in each row

    return ans  # return the final answer 