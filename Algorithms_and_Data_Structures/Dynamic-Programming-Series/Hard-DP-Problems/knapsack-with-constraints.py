def knapsack(weights, values, capacity):
    n = len(weights)
    # Create a DP array with (n+1) rows for items and (capacity+1) columns for capacity
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    # Fill the DP table
    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            if weights[i - 1] <= w:
                # Include the item or exclude it, choose the better option
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - weights[i - 1]] + values[i - 1])
            else:
                # Cannot include the item, so carry forward the value without it
                dp[i][w] = dp[i - 1][w]
    
    return dp[n][capacity]

# Test the function
weights = [1, 2, 3, 5]
values = [10, 20, 30, 50]
capacity = 6
print(f"Maximum value in the knapsack: {knapsack(weights, values, capacity)}")
