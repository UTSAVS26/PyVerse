"""
Buy and Sell Stocks With Transaction Fee
You may complete as many transactions as you like, but you must pay a transaction fee for each transaction.
"""

class Solution:
    def maxProfit(self, prices, fee):
        n = len(prices)  # get the number of days
        hold = -prices[0]  # max profit if holding a stock on day 0
        cash = 0  # max profit if not holding a stock
        for i in range(1, n):  # loop through each day
            hold = max(hold, cash - prices[i])  # either keep holding or buy today
            cash = max(cash, hold + prices[i] - fee)  # either keep cash or sell today (pay fee)
        return cash  # return the max profit at the end 