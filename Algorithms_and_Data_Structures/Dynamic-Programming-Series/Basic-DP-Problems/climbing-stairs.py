def climbStairs(n):
    # Handle base cases
    if n == 1:
        return 1
    if n == 2:
        return 2

    # Create a table to store the number of ways to reach each step
    dp = [0] * (n + 1)
    dp[1] = 1  # 1 way to reach the first step
    dp[2] = 2  # 2 ways to reach the second step

    # Fill the table in a bottom-up manner
    for i in range(3, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]

    return dp[n]

# Test the function
n = 5
print(f"Number of ways to climb {n} stairs: {climbStairs(n)}")
