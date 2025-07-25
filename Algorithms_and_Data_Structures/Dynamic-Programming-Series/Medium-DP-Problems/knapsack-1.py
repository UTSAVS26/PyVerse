"""
knapsack - 1
classic 0/1 knapsack problem: given weights and values, and a max weight, find the max value you can carry.
"""

class Solution:
    def knapsack(self, weights, values, max_weight):
        n = len(weights)  # get the number of items
        # dp[i][w] = max value using first i items with weight limit w
        dp = [[0] * (max_weight + 1) for _ in range(n + 1)]  # create a dp table to store results
        for i in range(1, n + 1):  # loop through each item
            for w in range(max_weight + 1):  # loop through each possible weight
                dp[i][w] = dp[i - 1][w]  # if we don't take this item, value stays the same as before
                if weights[i - 1] <= w:  # check if the current item can fit in the knapsack
                    # if we take this item, add its value and check the best value for the remaining weight
                    dp[i][w] = max(dp[i][w], values[i - 1] + dp[i - 1][w - weights[i - 1]])
        return dp[n][max_weight]  # return the maximum value we can get with all items and max weight 