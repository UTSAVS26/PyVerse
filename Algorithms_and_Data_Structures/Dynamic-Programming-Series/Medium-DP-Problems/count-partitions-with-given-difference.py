"""
Count Partitions with Given Difference
Given an array and a difference, count the number of ways to partition the array into two subsets such that the difference of their sums is equal to the given difference.
"""

class Solution:
    def countPartitions(self, nums, diff):
        total = sum(nums)  # get the total sum of the array
        if (total + diff) % 2 != 0:  # check if the sum can be split as required
            return 0  # if not, return 0
        target = (total + diff) // 2  # calculate the target sum for one subset
        n = len(nums)  # get the number of elements
        dp = [[0] * (target + 1) for _ in range(n + 1)]  # create a dp table
        for i in range(n + 1):  # base case: one way to get sum 0
            dp[i][0] = 1
        for i in range(1, n + 1):  # loop through each element
            for j in range(target + 1):  # loop through each possible sum
                if nums[i-1] <= j:  # if current number can be included
                    # include or exclude the current number
                    dp[i][j] = dp[i-1][j] + dp[i-1][j-nums[i-1]]
                else:
                    dp[i][j] = dp[i-1][j]  # exclude the current number
        return dp[n][target]  # return the number of ways 