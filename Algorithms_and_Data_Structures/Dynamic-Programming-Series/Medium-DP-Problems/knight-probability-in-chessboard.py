"""
Knight Probability in Chessboard
Given a n x n chessboard, a knight starts at the cell (row, column) and attempts to make exactly k moves. Each move, the knight chooses one of eight possible moves uniformly at random (even if the piece would go off the board) and moves there. The probability that the knight remains on the board after it has stopped moving is returned.
"""

class Solution:
    def knightProbability(self, n: int, k: int, row: int, column: int) -> float:
        # dp[move][i][j] = probability of knight at position (i, j) after 'move' moves
        dp = [[[0.0 for _ in range(n)] for _ in range(n)] for _ in range(k+1)]
        dp[0][row][column] = 1.0
        directions = [(2,1), (1,2), (-1,2), (-2,1), (-2,-1), (-1,-2), (1,-2), (2,-1)]
        for move in range(k):
            for i in range(n):
                for j in range(n):
                    if dp[move][i][j] > 0:
                        for dx, dy in directions:
                            ni, nj = i + dx, j + dy
                            if 0 <= ni < n and 0 <= nj < n:
                                dp[move+1][ni][nj] += dp[move][i][j] / 8.0
        # sum probabilities after k moves
        return sum(dp[k][i][j] for i in range(n) for j in range(n)) 