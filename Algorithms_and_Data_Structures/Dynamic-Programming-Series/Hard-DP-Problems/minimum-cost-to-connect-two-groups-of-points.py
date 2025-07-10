"""
Minimum Cost to Connect Two Groups of Points
Given two groups of points and a cost matrix, connect each point in both groups so that every point is connected to at least one point in the other group. Find the minimum total cost.
"""

class Solution:
    def connectTwoGroups(self, cost):
        m, n = len(cost), len(cost[0])
        dp = [float('inf')] * (1 << n)  # dp[mask] = min cost to connect first i rows and mask columns
        dp[0] = 0  # No columns connected yet
        for i in range(m):
            new_dp = [float('inf')] * (1 << n)
            for mask in range(1 << n):
                if dp[mask] == float('inf'):
                    continue
                for j in range(n):
                    new_mask = mask | (1 << j)
                    new_dp[new_mask] = min(new_dp[new_mask], dp[mask] + cost[i][j])
            dp = new_dp
        # After connecting all rows, make sure every column is connected
        for mask in range(1 << n):
            if dp[mask] < float('inf'):
                for j in range(n):
                    if not (mask & (1 << j)):
                        dp[mask | (1 << j)] = min(dp[mask | (1 << j)], dp[mask] + min(cost[i][j] for i in range(m)))
        return dp[(1 << n) - 1] 