# 0/1 Knapsack Problem Solution

# Problem: Given n items with weight and value, find the maximum value you can carry in a knapsack of capacity W.

def knapsack(weights, values, W):
    n = len(weights)
    # Create a 2D array to store maximum values
    dp = [[0] * (W + 1) for _ in range(n + 1)]

    # Fill the dp array
    for i in range(1, n + 1):
        for w in range(W + 1):
            if weights[i - 1] <= w:
                # Maximize value for the current item
                dp[i][w] = max(dp[i - 1][w], values[i - 1] + dp[i - 1][w - weights[i - 1]])
            else:
                dp[i][w] = dp[i - 1][w]

    return dp[n][W]

# Example usage
weights = [1, 2, 3]
values = [10, 20, 30]
W = 5
print(f"Maximum value in Knapsack: {knapsack(weights, values, W)}")  # Output: Maximum value in Knapsack: 50
