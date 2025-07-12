"""
Buy and Sell Stock - II
You may complete as many transactions as you like (i.e., buy one and sell one share of the stock multiple times) with the constraint that you may not engage in multiple transactions at the same time.
"""

class Solution:
    def maxProfit(self, prices):
        profit = 0  # start with zero profit
        for i in range(1, len(prices)):  # loop through each day starting from the second
            if prices[i] > prices[i-1]:  # if today's price is higher than yesterday's
                profit += prices[i] - prices[i-1]  # buy yesterday, sell today
        return profit  # return the total profit 