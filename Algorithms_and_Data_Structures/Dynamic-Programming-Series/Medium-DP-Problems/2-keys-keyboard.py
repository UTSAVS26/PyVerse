"""
2 keys keyboard
given n, find the minimum number of steps to get n 'a's on a notepad using only copy all and paste operations.
"""

class Solution:
    def minSteps(self, n):
        # dp[i] = min steps to get i 'a's
        dp = [0] * (n + 1)  # create a dp array to store min steps for each number of 'a's
        for i in range(2, n + 1):  # start from 2 because 1 'a' needs 0 steps
            dp[i] = i  # initialize with the worst case (all pastes)
            for j in range(2, int(i ** 0.5) + 1):  # try all possible divisors
                if i % j == 0:  # if j is a divisor of i
                    # try to build i by first getting j 'a's, then pasting (i // j) times
                    dp[i] = min(dp[i], dp[j] + i // j)
                    # also try the other divisor
                    dp[i] = min(dp[i], dp[i // j] + j)
        return dp[n]  # return the minimum steps to get n 'a's 