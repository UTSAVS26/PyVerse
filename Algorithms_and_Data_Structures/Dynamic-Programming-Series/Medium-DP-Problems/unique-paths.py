def uniquePaths(m, n):
    dp = [[1] * n for _ in range(m)]

    # Fill the dp array using the recurrence relation:
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i - 1][j] + dp[i][j - 1]

    # The bottom-right corner contains the total number of unique paths
    return dp[-1][-1]

m = 3
n = 7
print(f"Number of unique paths for a {m}x{n} grid: {uniquePaths(m, n)}")
