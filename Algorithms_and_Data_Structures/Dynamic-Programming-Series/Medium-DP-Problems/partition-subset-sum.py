"""
partition subset sum
given an array, find if there is a subset with sum equal to a given value.
"""

def canPartition(nums):
    total_sum = sum(nums)
    
    # If the total sum is odd, we can't partition it into two equal subsets
    if total_sum % 2 != 0:
        return False
    
    target = total_sum // 2
    n = len(nums)
    
    # DP array to store whether a specific sum is possible
    dp = [False] * (target + 1)
    dp[0] = True
    
    for num in nums:
        for i in range(target, num - 1, -1):
            dp[i] = dp[i] or dp[i - num]
    
    # If dp[target] is True, then it's possible to partition the array
    return dp[target]

nums = [1, 5, 11, 5]
print(f"Can partition: {canPartition(nums)}")

class Solution:
    def isSubsetSum(self, nums, target):
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
