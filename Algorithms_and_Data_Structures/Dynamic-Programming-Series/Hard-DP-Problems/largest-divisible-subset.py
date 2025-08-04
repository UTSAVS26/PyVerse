"""
largest divisible subset
given a set of distinct positive integers, find the largest subset such that every pair (si, sj) of elements in this subset satisfies: si % sj == 0 or sj % si == 0.
"""

class Solution:
    def largestDivisibleSubset(self, nums):
        if not nums:  # check if the list is empty
            return []  # return empty list if no numbers
        nums.sort()  # sort the numbers to make divisibility checks easier
        n = len(nums)  # get the number of elements
        dp = [1] * n  # dp[i] = size of largest subset ending at i
        prev = [-1] * n  # prev[i] = index of previous element in the subset
        max_idx = 0  # index of the largest subset's last element
        for i in range(n):  # loop through each number
            for j in range(i):  # check all previous numbers
                if nums[i] % nums[j] == 0 and dp[j] + 1 > dp[i]:  # check divisibility and if it makes a bigger subset
                    dp[i] = dp[j] + 1  # update the size of the subset
                    prev[i] = j  # update the previous index
            if dp[i] > dp[max_idx]:  # update the index of the largest subset
                max_idx = i
        res = []  # list to store the result
        while max_idx != -1:  # reconstruct the subset
            res.append(nums[max_idx])  # add the number to the result
            max_idx = prev[max_idx]  # move to the previous index
        return res[::-1]  # reverse the result to get the correct order 