"""
last stone weight ii
given stones with weights, split them into two groups so the difference of their sums is minimized.
"""

class Solution:
    def lastStoneWeightII(self, stones):
        total = sum(stones)  # get the total weight of all stones
        n = len(stones)  # get the number of stones
        target = total // 2  # target is half the total weight
        dp = [0] * (target + 1)  # create a dp array for possible weights
        for stone in stones:  # loop through each stone
            for j in range(target, stone - 1, -1):  # loop backwards for each possible weight
                # try to put this stone in one group and update the best sum
                dp[j] = max(dp[j], dp[j - stone] + stone)
        return total - 2 * dp[target]  # return the minimum possible difference 