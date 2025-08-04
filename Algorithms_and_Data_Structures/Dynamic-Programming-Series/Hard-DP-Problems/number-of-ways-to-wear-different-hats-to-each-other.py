"""
number of ways to wear different hats to each other
given n people and 40 types of hats, each person has a list of hats they like. find the number of ways to assign hats so that each person wears one hat they like and no two people wear the same hat.
"""

class Solution:
    def numberWays(self, hats):
        n = len(hats)  # get the number of people
        hat_to_people = [[] for _ in range(41)]  # hats are 1-indexed
        for person, hat_list in enumerate(hats):  # loop through each person
            for hat in hat_list:  # loop through each hat the person likes
                hat_to_people[hat].append(person)  # add the person to the list for this hat
        dp = [0] * (1 << n)  # dp[mask] = number of ways to assign hats for this mask
        dp[0] = 1  # no one has a hat yet
        for hat in range(1, 41):  # try each hat
            new_dp = dp[:]  # copy the current dp
            for mask in range(1 << n):  # loop through all possible assignments
                if dp[mask] == 0:  # if there are no ways for this mask, skip
                    continue
                for person in hat_to_people[hat]:  # try to give this hat to each person who likes it
                    if not (mask & (1 << person)):  # if this person doesn't have a hat yet
                        new_mask = mask | (1 << person)  # assign the hat to this person
                        new_dp[new_mask] += dp[mask]  # add the number of ways
                        new_dp[new_mask] %= 10**9 + 7  # to avoid large numbers
            dp = new_dp  # update dp for the next hat
        return dp[(1 << n) - 1]  # return the number of ways to assign hats to everyone 