"""
Partition to K Equal Sum Subsets
Given an array of positive integers and an integer k, find out if it's possible to divide the array into k non-empty subsets whose sums are all equal.
"""

class Solution:
    def canPartitionKSubsets(self, nums, k):
        # first, check if the total sum is divisible by k
        total = sum(nums)
        if total % k != 0:
            return False  # if not divisible, it's impossible
        target = total // k  # each subset must sum to this value
        nums.sort(reverse=True)  # sort numbers in descending order for efficiency
        used = [False] * len(nums)  # track which numbers are already used
        
        # helper function to try to fill each subset
        def backtrack(start, k, current_sum):
            # if only one subset left, the rest must form the last subset
            if k == 1:
                return True
            # if current subset is filled, move to next subset
            if current_sum == target:
                return backtrack(0, k - 1, 0)
            for i in range(start, len(nums)):
                if not used[i] and current_sum + nums[i] <= target:
                    used[i] = True  # mark this number as used
                    if backtrack(i + 1, k, current_sum + nums[i]):
                        return True  # found a valid way
                    used[i] = False  # backtrack if not valid
            return False  # no valid way found
        
        return backtrack(0, k, 0) 