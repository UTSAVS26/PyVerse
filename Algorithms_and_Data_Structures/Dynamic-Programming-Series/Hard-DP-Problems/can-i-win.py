"""
Can I Win
This is a game where two players take turns picking unique numbers from 1 to maxChoosableInteger. Each number can only be picked once. The player whose total reaches or exceeds the desiredTotal first wins. Determine if the first player can guarantee a win if both play optimally.
"""

class Solution:
    def canIWin(self, maxChoosableInteger, desiredTotal):
        # if the sum of all numbers is less than the target, it's impossible to win
        if (maxChoosableInteger * (maxChoosableInteger + 1)) // 2 < desiredTotal:
            return False  # not enough total to reach the target
        
        # use a dictionary to remember already computed states
        memo = {}
        
        # helper function for recursive DP
        def can_win(used, total):
            # If we've already solved this state, return the answer
            if used in memo:
                return memo[used]
            
            # try every number that hasn't been used yet
            for i in range(1, maxChoosableInteger + 1):
                if not (used & (1 << i)):
                    # if picking this number wins the game, or forces the opponent to lose
                    if total + i >= desiredTotal or not can_win(used | (1 << i), total + i):
                        memo[used] = True  # this move guarantees a win
                        return True
            memo[used] = False  # no move can guarantee a win from this state
            return False
        
        # start the game with no numbers used and total at 0
        return can_win(0, 0) 