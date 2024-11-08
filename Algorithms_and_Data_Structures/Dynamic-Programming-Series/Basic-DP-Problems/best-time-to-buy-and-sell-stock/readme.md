# Best Time To Buy And Sell Stock

## Problem Statement
Given an array `prices` where `prices[i]` represents the price of a given stock on the ith day, you need to maximize your profit by selecting a single day to buy the stock and a different day in the future to sell it. 

- The goal is to determine the maximum profit you can achieve from this single transaction.
- If no profit is possible (i.e., prices decrease continuously), return 0.

### Example
1. **Input**: `prices = [7, 1, 5, 3, 6, 4]`
   - **Output**: `5`
   - **Explanation**: Buy on day 2 (price = 1) and sell on day 5 (price = 6), achieving a maximum profit of `6 - 1 = 5`.

2. **Input**: `prices = [7, 6, 4, 3, 1]`
   - **Output**: `0`
   - **Explanation**: No profit can be made, as prices are in descending order.

## Solution Approach
The solution involves a single-pass algorithm to find the maximum profit efficiently.

1. **Track Minimum Price**: Keep track of the lowest stock price encountered as you iterate over the array. This is the day to "buy" for maximum profit.
2. **Calculate Maximum Profit**: For each price, calculate the potential profit if you were to sell on that day by subtracting the minimum price seen so far.
3. **Update Maximum Profit**: If the potential profit for the current price is greater than the maximum profit so far, update the maximum profit.

This approach has a time complexity of **O(n)**, as we only traverse the `prices` array once.
