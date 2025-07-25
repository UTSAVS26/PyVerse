"""
Distinct Subsequences
Given two strings s and t, return the number of distinct subsequences of s which equals t.
"""

class Solution:
    def numDistinct(self, s, t):
        m, n = len(s), len(t)  # get the lengths of s and t
        # dp[i][j] = number of ways s[i:] can form t[j:]
        dp = [[0] * (n + 1) for _ in range(m + 1)]  # create a dp table
        # if t is empty, there is one way (delete all from s)
        for i in range(m + 1):
            dp[i][n] = 1
        # fill the table from the end
        for i in range(m - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                if s[i] == t[j]:
                    # we can match s[i] with t[j] or skip s[i]
                    dp[i][j] = dp[i + 1][j + 1] + dp[i + 1][j]
                else:
                    # skip s[i]
                    dp[i][j] = dp[i + 1][j]
        return dp[0][0]  # return the number of distinct subsequences 