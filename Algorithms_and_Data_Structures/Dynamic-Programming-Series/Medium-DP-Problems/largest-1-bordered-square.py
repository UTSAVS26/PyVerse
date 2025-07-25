"""
Problem: Largest 1-Bordered Square
Difficulty: Medium

This problem gives you a grid of 0's and 1's. You need to find the size of the largest square whose border is made up of only 1's.

Approach:
For each cell, precompute the number of consecutive 1's to the left and up. For each possible bottom-right corner, check the largest possible square ending there by verifying if the top and left borders are also all 1's. Keep track of the largest size found.
"""

def largest1BorderedSquare(grid):
    m, n = len(grid), len(grid[0])  # Get the dimensions of the grid
    hor = [[0]*n for _ in range(m)]  # Horizontal consecutive 1's
    ver = [[0]*n for _ in range(m)]  # Vertical consecutive 1's

    # Precompute horizontal and vertical consecutive 1's for each cell
    for i in range(m):
        for j in range(n):
            if grid[i][j]:
                # If current cell is 1, add 1 to the left and up counts
                hor[i][j] = (hor[i][j-1] if j else 0) + 1
                ver[i][j] = (ver[i-1][j] if i else 0) + 1
            # If current cell is 0, both counts remain 0

    max_side = 0  # To keep track of the largest square's side length
    # Try every cell as the bottom-right corner of a square
    for i in range(m):
        for j in range(n):
            # The largest possible square ending at (i, j)
            small = min(hor[i][j], ver[i][j])
            while small > 0:
                # Check if the top and left borders are also all 1's
                if hor[i-small+1][j] >= small and ver[i][j-small+1] >= small:
                    max_side = max(max_side, small)  # Update max_side
                    break  # No need to check smaller squares
                small -= 1  # Try a smaller square
    return max_side * max_side  # Return the area of the largest square 