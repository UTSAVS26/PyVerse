"""
Target Sum
Given an array of integers and a target, count the number of ways to assign + and - signs to make the sum equal to target.
"""

class Solution:
    def findTargetSumWays(self, nums, target):
        total = sum(nums)
        if (total + target) % 2 != 0 or total < abs(target):
            return 0
        s1 = (total + target) // 2
        n = len(nums)
        dp = [[0] * (s1 + 1) for _ in range(n + 1)]
        dp[0][0] = 1
        for i in range(1, n + 1):
            for j in range(s1 + 1):
                if nums[i-1] <= j:
                    dp[i][j] = dp[i-1][j] + dp[i-1][j-nums[i-1]]
                else:
                    dp[i][j] = dp[i-1][j]
        return dp[n][s1] 