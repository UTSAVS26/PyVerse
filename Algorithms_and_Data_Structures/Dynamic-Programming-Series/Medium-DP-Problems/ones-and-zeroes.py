"""
ones and zeroes
given a list of binary strings and m 0's and n 1's, find the max number of strings you can pick with at most m 0's and n 1's.
"""

class Solution:
    def findMaxForm(self, strs, m, n):
        # dp[i][j] = max number of strings with i 0's and j 1's
        dp = [[0] * (n + 1) for _ in range(m + 1)]  # create a dp table for 0's and 1's
        for s in strs:  # loop through each string
            zeros = s.count('0')  # count number of 0's in the string
            ones = s.count('1')  # count number of 1's in the string
            for i in range(m, zeros - 1, -1):  # loop backwards for 0's
                for j in range(n, ones - 1, -1):  # loop backwards for 1's
                    # check if we can include this string and update the dp value
                    dp[i][j] = max(dp[i][j], 1 + dp[i - zeros][j - ones])
        return dp[m][n]  # return the max number of strings we can pick 