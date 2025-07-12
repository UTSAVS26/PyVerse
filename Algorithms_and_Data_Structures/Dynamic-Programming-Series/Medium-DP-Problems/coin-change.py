"""
coin change
given coins and an amount, find the minimum number of coins needed to make up that amount. if not possible, return -1.
"""

class Solution:
    def coinChange(self, coins, amount):
        # dp[i] = min coins to make amount i
        dp = [float('inf')] * (amount + 1)  # create a dp array to store min coins for each amount
        dp[0] = 0  # base case: 0 coins needed for amount 0
        for coin in coins:  # loop through each coin
            for i in range(coin, amount + 1):  # loop through each amount from coin to amount
                # check if using this coin gives a better result
                dp[i] = min(dp[i], dp[i - coin] + 1)
        return dp[amount] if dp[amount] != float('inf') else -1  # return result or -1 if not possible


coins = [1, 2, 5]
amount = 11
print(f"Fewest number of coins to make {amount}: {Solution().coinChange(coins, amount)}")
