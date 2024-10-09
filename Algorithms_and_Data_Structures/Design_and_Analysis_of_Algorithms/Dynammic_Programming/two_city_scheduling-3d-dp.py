"""
Problem Statement:
You are given two cities, A and B, and a list of people, where each person has a cost associated with flying to either city.
Your goal is to find the optimal way to send N people to city A and N people to city B such that the total cost is minimized.

The input consists of an array 'costs' where 'costs[i] = [aCost, bCost]' represents the cost of flying the i-th person to city A and city B.

Objective:
Return the minimum total cost to send N people to each city.

Example:
Input: costs = [[10, 20], [30, 200], [50, 30], [200, 500]]
Output: 370

Constraints:
- 2 * N == costs.length
- 2 <= costs.length <= 100
- costs.length is even.
- 1 <= aCosti, bCosti <= 1000
"""

from typing import List
import sys

class Solution:
    def twoCitySchedCost(self, costs: List[List[int]]) -> int:
        N = len(costs) // 2
        
        # DP table to store minimum costs
        dp = [[[0] * (N + 1) for _ in range(N + 1)] for _ in range(2 * N + 1)]
        
        # Initialize the DP table for base case where no person is selected
        for i in range(N + 1):
            for j in range(N + 1):
                dp[0][i][j] = 0  # No cost if no person is selected
        
        # Iterate through each person
        for i in range(1, 2 * N + 1):
            # Send i-th person to city A
            for wA in range(1, N + 1):
                if dp[i - 1][wA - 1][0] == sys.maxsize:
                    dp[i][wA][0] = sys.maxsize  # Handle overflow cases
                else:
                    dp[i][wA][0] = costs[i - 1][0] + dp[i - 1][wA - 1][0]

            # Send i-th person to city B
            for wB in range(1, N + 1):
                if dp[i - 1][0][wB - 1] == sys.maxsize:
                    dp[i][0][wB] = sys.maxsize  # Handle overflow cases
                else:
                    dp[i][0][wB] = costs[i - 1][1] + dp[i - 1][0][wB - 1]
            
            dp[i][0][0] = sys.maxsize  # Set invalid base case
        
        # Main DP transition for assigning each person to either city
        for i in range(1, 2 * N + 1):
            for wA in range(1, N + 1):
                for wB in range(1, N + 1):
                    if dp[i - 1][wA - 1][wB] == sys.maxsize:
                        dp[i][wA][wB] = costs[i - 1][1]
                    elif dp[i - 1][wA][wB - 1] == sys.maxsize:
                        dp[i][wA][wB] = costs[i - 1][0]
                    else:
                        dp[i][wA][wB] = min(costs[i - 1][0] + dp[i - 1][wA - 1][wB],
                                            costs[i - 1][1] + dp[i - 1][wA][wB - 1])
        
        # Return the minimum cost of assigning N people to city A and N people to city B
        return dp[2 * N][N][N]

"""
Dry Run Example:
Input: costs = [[10, 20], [30, 200], [50, 30], [200, 500]]

N = 2 (because there are 4 people in total, so 2 people should go to city A and 2 to city B).

1. Initialize DP table:
    dp[i][wA][wB] will represent the minimum cost when considering the first 'i' people, 
    where 'wA' people are sent to city A and 'wB' people are sent to city B.

2. Start filling DP table:

Iteration 1 (i = 1):
- Person 1: costs = [10, 20]
- We can either send them to city A or city B:
    a. If sent to city A, dp[1][1][0] = 10
    b. If sent to city B, dp[1][0][1] = 20

Iteration 2 (i = 2):
- Person 2: costs = [30, 200]
- Two possibilities:
    a. If sent to city A, and 1 person is already sent to city A, dp[2][2][0] = dp[1][1][0] + 30 = 10 + 30 = 40
    b. If sent to city B, and 1 person is already sent to city A, dp[2][1][1] = dp[1][1][0] + 200 = 10 + 200 = 210

Iteration 3 (i = 3):
- Person 3: costs = [50, 30]
- Again, two possibilities:
    a. Send to city A: dp[3][2][1] = dp[2][1][1] + 50 = 210 + 50 = 260
    b. Send to city B: dp[3][1][2] = dp[2][1][1] + 30 = 210 + 30 = 240

Iteration 4 (i = 4):
- Person 4: costs = [200, 500]
- Two possibilities:
    a. Send to city A: dp[4][2][2] = dp[3][1][2] + 200 = 240 + 200 = 440
    b. Send to city B: dp[4][2][2] = dp[3][2][1] + 500 = 260 + 500 = 760
    The minimum is 440.

3. Final result: The minimum cost to send 2 people to each city is 440.

"""
