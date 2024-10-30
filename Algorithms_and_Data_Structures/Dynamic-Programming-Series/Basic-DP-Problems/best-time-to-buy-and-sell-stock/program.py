from typing import List

class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        # Initialize the minimum price as the price on the first day
        min_price = prices[0]
        # Initialize max profit as 0 (because we can't make a profit if prices always go down)
        max_profit = 0

        # Loop through the prices starting from the second day
        for i in range(1, len(prices)):
            # Calculate the profit if we were to sell on this day
            profit_today = prices[i] - min_price
            # Update max profit if the profit today is higher
            max_profit = max(max_profit, profit_today)
            # Update min_price to be the lowest price seen so far
            min_price = min(min_price, prices[i])

        return max_profit

# Example usage
solution = Solution()

# Example 1
prices1 = [7, 1, 5, 3, 6, 4]
print("Input:", prices1)
print("Max Profit:", solution.maxProfit(prices1))  # Expected output: 5

# Example 2
prices2 = [7, 6, 4, 3, 1]
print("\nInput:", prices2)
print("Max Profit:", solution.maxProfit(prices2))  # Expected output: 0 (no profit possible)
