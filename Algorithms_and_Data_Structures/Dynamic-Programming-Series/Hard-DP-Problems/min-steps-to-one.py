def min_steps_to_one(n):
    # Initialize a DP array where dp[i] will store minimum steps to reach 1 from i
    dp = [float('inf')] * (n + 1)
    dp[1] = 0  # Base case: it takes 0 steps to reach 1 from 1
    
    # Fill the DP table for each number from 2 up to n
    for i in range(2, n + 1):
        # Subtract 1
        dp[i] = dp[i - 1] + 1
        # Divide by 2 if applicable
        if i % 2 == 0:
            dp[i] = min(dp[i], dp[i // 2] + 1)
        # Divide by 3 if applicable
        if i % 3 == 0:
            dp[i] = min(dp[i], dp[i // 3] + 1)
    
    return dp[n]

# Test the function
n = 10
print(f"Minimum steps to reduce {n} to 1: {min_steps_to_one(n)}")
