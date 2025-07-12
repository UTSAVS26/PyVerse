"""
count subsets with sum k
given an array of integers and a target sum k, count the number of subsets whose sum is exactly k.
"""

class Solution:
    def countSubsets(self, nums, k):
        n = len(nums)  # get the number of elements
        dp = [[0] * (k + 1) for _ in range(n + 1)]  # create a dp table
        for i in range(n + 1):  # base case: one way to get sum 0
            dp[i][0] = 1
        for i in range(1, n + 1):  # loop through each element
            for j in range(k + 1):  # loop through each possible sum
                if nums[i-1] <= j:  # if current number can be included
                    # include or exclude the current number
                    dp[i][j] = dp[i-1][j] + dp[i-1][j-nums[i-1]]
                else:
                    dp[i][j] = dp[i-1][j]  # exclude the current number
        return dp[n][k]  # return the number of subsets 