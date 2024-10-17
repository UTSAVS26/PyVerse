def coinChange(coins, amount):
    # Initialize the dp array with a value larger than the possible answer (amount + 1)
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0  

    for i in range(1, amount + 1):
        for coin in coins:
            if i - coin >= 0:  
                dp[i] = min(dp[i], dp[i - coin] + 1)  

    # If dp[amount] is still infinity, it means the amount cannot be made up
    return dp[amount] if dp[amount] != float('inf') else -1


coins = [1, 2, 5]
amount = 11
print(f"Fewest number of coins to make {amount}: {coinChange(coins, amount)}")
