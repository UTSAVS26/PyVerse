"""
Probability of Two Boxes Having the Same Number of Distinct Balls
Given 2n balls of k different colors, find the probability that when the balls are randomly divided into two boxes with n balls each, both boxes have the same number of distinct colors.
"""

from math import comb
from functools import lru_cache

class Solution:
    def getProbability(self, balls):
        n = sum(balls) // 2  # total balls in each box
        k = len(balls)  # number of colors
        total = 0  # total number of ways
        valid = 0  # number of valid ways
        @lru_cache(None)
        def dfs(i, a, b, ca, cb):
            nonlocal total, valid
            if i == k:  # if all colors are processed
                if a == b == n and ca == cb:  # if both boxes have n balls and same number of colors
                    valid += 1  # count as valid
                if a == b == n:  # if both boxes have n balls
                    total += 1  # count as total
                return
            for x in range(balls[i]+1):  # try all possible splits for this color
                y = balls[i] - x  # remaining balls go to the other box
                dfs(i+1, a+x, b+y, ca+(x>0), cb+(y>0))  # recurse for next color
        dfs(0,0,0,0,0)  # start dfs with 0 balls and 0 colors in both boxes
        return valid/total if total else 0  # return the probability 