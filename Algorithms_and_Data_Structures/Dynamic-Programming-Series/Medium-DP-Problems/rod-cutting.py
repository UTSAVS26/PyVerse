"""
rod cutting
given a rod of length n and prices for each length, find the max value you can get by cutting the rod and selling the pieces.
"""

class Solution:
    def cutRod(self, price, n):
        # dp[i] = max value for rod of length i
        dp = [0] * (n + 1)  # create a dp array to store max value for each length
        for i in range(1, n + 1):  # loop through each possible rod length
            for j in range(i):  # try every possible first cut
                # update the max value for length i by cutting at position j
                dp[i] = max(dp[i], price[j] + dp[i - j - 1])
        return dp[n]  # return the max value for the full rod 