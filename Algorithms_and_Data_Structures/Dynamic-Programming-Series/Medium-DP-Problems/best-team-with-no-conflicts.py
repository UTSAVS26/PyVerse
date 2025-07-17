"""
best team with no conflicts
given scores and ages, find the best team score with no conflicts (no younger player has a higher score than an older player).
"""

class Solution:
    def bestTeamScore(self, scores, ages):
        players = sorted(zip(ages, scores))  # pair ages and scores, then sort by age
        n = len(scores)  # get the number of players
        dp = [0] * n  # create a dp array to store best scores
        for i in range(n):  # loop through each player
            dp[i] = players[i][1]  # start with this player's score
            for j in range(i):  # check all previous players
                if players[j][1] <= players[i][1]:  # if no conflict in score
                    dp[i] = max(dp[i], dp[j] + players[i][1])  # update best score
        return max(dp)  # return the best team score 