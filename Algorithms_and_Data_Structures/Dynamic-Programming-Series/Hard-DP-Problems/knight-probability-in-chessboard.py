"""
knight probability in chessboard
given a n x n chessboard, a knight starts at the cell (row, column) and attempts to make exactly k moves. each move, the knight chooses one of eight possible moves uniformly at random (even if the piece would go off the board) and moves there. the probability that the knight remains on the board after it has stopped moving is returned.
"""

class Solution:
    def knightProbability(self, n: int, k: int, row: int, column: int) -> float:
        dp = [[[0.0 for _ in range(n)] for _ in range(n)] for _ in range(k+1)]  # create a 3d dp array for moves and positions
        dp[0][row][column] = 1.0  # start with probability 1 at the initial position
        directions = [(2,1), (1,2), (-1,2), (-2,1), (-2,-1), (-1,-2), (1,-2), (2,-1)]  # all possible knight moves
        for move in range(k):  # loop through each move
            for i in range(n):  # loop through each row
                for j in range(n):  # loop through each column
                    if dp[move][i][j] > 0:  # if the knight can be at (i, j) after 'move' moves
                        for dx, dy in directions:  # try all possible moves
                            ni, nj = i + dx, j + dy  # calculate new position
                            if 0 <= ni < n and 0 <= nj < n:  # check if new position is on the board
                                dp[move+1][ni][nj] += dp[move][i][j] / 8.0  # update probability for the new position
        return sum(dp[k][i][j] for i in range(n) for j in range(n))  # sum probabilities after k moves 