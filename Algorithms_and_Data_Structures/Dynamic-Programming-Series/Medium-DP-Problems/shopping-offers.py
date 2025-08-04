"""
shopping offers
given price list, special offers, and needs, find the minimum cost to satisfy all needs.
"""

class Solution:
    def shoppingOffers(self, price, special, needs):
        memo = {}  # use a memo dictionary to store results for each state
        def dfs(cur):  # define a helper function for dfs
            key = tuple(cur)  # use the current needs as the key
            if key in memo:  # if we've already solved this state
                return memo[key]  # return the stored result
            res = sum(cur[i] * price[i] for i in range(len(cur)))  # buy everything without offers
            for offer in special:  # try each special offer
                nxt = [cur[i] - offer[i] for i in range(len(cur))]  # apply the offer
                if min(nxt) >= 0:  # check if the offer is valid
                    res = min(res, offer[-1] + dfs(nxt))  # take the offer and solve for the rest
            memo[key] = res  # store the result
            return res  # return the minimum cost
        return dfs(needs)  # start the dfs with the initial needs 