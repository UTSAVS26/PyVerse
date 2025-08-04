"""
Buy and Sell Stocks With Cooldown
You may complete as many transactions as you like, but after you sell your stock, you cannot buy stock on the next day (i.e., cooldown one day).
"""

class Solution:
    def maxProfit(self, prices):
        if not prices:  # check if the prices list is empty
            return 0  # no profit if there are no prices
        n = len(prices)  # get the number of days
        hold = [0]*n  # hold[i] = max profit if holding a stock on day i
        sold = [0]*n  # sold[i] = max profit if just sold a stock on day i
        rest = [0]*n  # rest[i] = max profit if in cooldown or doing nothing on day i
        hold[0] = -prices[0]  # buy stock on the first day
        for i in range(1, n):  # loop through each day
            hold[i] = max(hold[i-1], rest[i-1] - prices[i])  # either keep holding or buy today
            sold[i] = hold[i-1] + prices[i]  # sell the stock today
            rest[i] = max(rest[i-1], sold[i-1])  # either stay in rest or just finished cooldown
        return max(sold[-1], rest[-1])  # return the max profit at the end 