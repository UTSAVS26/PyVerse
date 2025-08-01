"""
knapsack - 2
unbounded knapsack: you can take any item as many times as you want. maximize value for max weight.
"""

class Solution:
    def unboundedKnapsack(self, weights, values, max_weight):
        n = len(weights)  # get the number of items
        # dp[w] = max value for weight w
        dp = [0] * (max_weight + 1)  # create a dp array to store results for each weight
        for w in range(max_weight + 1):  # loop through each possible weight
            for i in range(n):  # loop through each item
                if weights[i] <= w:  # check if the item can fit in the current weight
                    # if we take this item, add its value and check the best value for the remaining weight
                    dp[w] = max(dp[w], values[i] + dp[w - weights[i]])
        return dp[max_weight]  # return the maximum value for the max weight 