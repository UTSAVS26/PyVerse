"""
Partition Equal Subset Sum
Given a non-empty array, determine if it can be partitioned into two subsets with equal sum.
"""

class Solution:
    def canPartition(self, nums):
        total = sum(nums)  # get the total sum of the array
        if total % 2 != 0:  # if the sum is odd, can't split equally
            return False  # return false
        target = total // 2  # target sum for each subset
        dp = [False] * (target + 1)  # create a dp array
        dp[0] = True  # base case: sum 0 is always possible
        for num in nums:  # loop through each number
            for i in range(target, num - 1, -1):  # loop backwards for each possible sum
                # check if we can form sum i by including this number
                dp[i] = dp[i] or dp[i - num]
        return dp[target]  # return true if target sum is possible 