"""
subset sum equal to target
given an array of integers and a target sum, determine if there is a subset of the array that sums to the target.
"""

class Solution:
    def subsetSum(self, nums, target):
        n = len(nums)  # get the number of elements
        dp = [[False] * (target + 1) for _ in range(n + 1)]  # create a dp table
        for i in range(n + 1):  # base case: sum 0 is always possible
            dp[i][0] = True
        for i in range(1, n + 1):  # loop through each element
            for j in range(1, target + 1):  # loop through each possible sum
                if nums[i-1] <= j:  # if current number can be included
                    # include or exclude the current number
                    dp[i][j] = dp[i-1][j] or dp[i-1][j-nums[i-1]]
                else:
                    dp[i][j] = dp[i-1][j]  # exclude the current number
        return dp[n][target]  # return true if target sum is possible 