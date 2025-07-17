"""
longest increasing subsequence
given an array, find the length of the longest increasing subsequence.
"""

class Solution:
    def lengthOfLIS(self, nums):
        if not nums:  # check if the array is empty
            return 0  # if empty, return 0
        n = len(nums)  # get the number of elements
        dp = [1] * n  # create a dp array, each element is at least a subsequence of length 1
        for i in range(n):  # loop through each element
            for j in range(i):  # check all previous elements
                if nums[i] > nums[j]:  # if current is greater, it can extend the subsequence
                    dp[i] = max(dp[i], dp[j] + 1)  # update the length
        return max(dp)  # return the maximum length found

nums = [10, 9, 2, 5, 3, 7, 101, 18]
print(f"Length of the longest increasing subsequence: {Solution().lengthOfLIS(nums)}")
