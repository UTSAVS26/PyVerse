"""
Buy and Sell Stock
Given an array for which the ith element is the price of a given stock on day i, design an algorithm to find the maximum profit. You may complete at most one transaction.
"""

class Solution:
    def maxProfit(self, prices):
        min_price = float('inf')  # set the minimum price to infinity initially
        max_profit = 0  # start with zero profit
        for price in prices:  # loop through each price
            min_price = min(min_price, price)  # update the minimum price so far
            max_profit = max(max_profit, price - min_price)  # update the max profit if selling today is better
        return max_profit  # return the maximum profit found 