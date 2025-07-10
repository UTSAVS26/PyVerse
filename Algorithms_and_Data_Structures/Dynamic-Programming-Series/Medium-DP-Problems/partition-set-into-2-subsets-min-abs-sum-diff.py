"""
Partition Set Into 2 Subsets With Minimum Absolute Sum Difference
Given an array, partition it into two subsets such that the absolute difference of their sums is minimized.
"""

class Solution:
    def minimumDifference(self, nums):
        total = sum(nums)  # get the total sum of the array
        n = len(nums)  # get the number of elements
        target = total // 2  # target is half the total sum
        dp = [False] * (target + 1)  # create a dp array
        dp[0] = True  # base case: sum 0 is always possible
        for num in nums:  # loop through each number
            for i in range(target, num - 1, -1):  # loop backwards for each possible sum
                # check if we can form sum i by including this number
                dp[i] = dp[i] or dp[i - num]
        for i in range(target, -1, -1):  # check from largest possible sum
            if dp[i]:  # if this sum is possible
                return abs(total - 2 * i)  # return the minimum difference 