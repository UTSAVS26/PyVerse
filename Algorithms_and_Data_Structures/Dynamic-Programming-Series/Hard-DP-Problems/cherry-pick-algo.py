def cherryPickup(grid):
    n = len(grid)
    # Initialize a 3D DP table with -inf to denote uncomputed values
    dp = [[[float('-inf')] * n for _ in range(n)] for _ in range(n)]
    dp[0][0][0] = grid[0][0]  # Starting position
    
    for r1 in range(n):
        for c1 in range(n):
            for r2 in range(n):
                c2 = r1 + c1 - r2
                if c2 < 0 or c2 >= n or grid[r1][c1] == -1 or grid[r2][c2] == -1:
                    continue
                
                cherries = grid[r1][c1]
                if r1 != r2:  
                    cherries += grid[r2][c2]
                
                if r1 > 0 and r2 > 0:
                    dp[r1][c1][r2] = max(dp[r1][c1][r2], dp[r1 - 1][c1][r2 - 1] + cherries)
                if r1 > 0 and c2 > 0:
                    dp[r1][c1][r2] = max(dp[r1][c1][r2], dp[r1 - 1][c1][r2] + cherries)
                if c1 > 0 and r2 > 0:
                    dp[r1][c1][r2] = max(dp[r1][c1][r2], dp[r1][c1 - 1][r2 - 1] + cherries)
                if c1 > 0 and c2 > 0:
                    dp[r1][c1][r2] = max(dp[r1][c1][r2], dp[r1][c1 - 1][r2] + cherries)
    
    return max(0, dp[n - 1][n - 1][n - 1])

grid = [
    [0, 1, -1],
    [1, 0, -1],
    [1, 1,  1]
]
print(f"Maximum cherries collected: {cherryPickup(grid)}")
