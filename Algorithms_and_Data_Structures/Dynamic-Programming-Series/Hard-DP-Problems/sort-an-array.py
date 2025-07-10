"""
Sort an Array
Given an array of integers, sort the array in ascending order using merge sort.
"""

class Solution:
    def sortArray(self, nums):
        # if the array has 1 or 0 elements, it's already sorted
        if len(nums) <= 1:
            return nums
        mid = len(nums) // 2  # find the middle index
        # sort the left and right halves
        left = self.sortArray(nums[:mid])
        right = self.sortArray(nums[mid:])
        # merge the sorted halves
        return self.merge(left, right)
    
    def merge(self, left, right):
        result = []  # create a list to store the merged result
        i = j = 0  # pointers for left and right lists
        # merge elements from left and right in order
        while i < len(left) and j < len(right):
            if left[i] < right[j]:
                result.append(left[i])  # add the smaller element
                i += 1
            else:
                result.append(right[j])  # add the smaller element
                j += 1
        # add any remaining elements
        result.extend(left[i:])
        result.extend(right[j:])
        return result 