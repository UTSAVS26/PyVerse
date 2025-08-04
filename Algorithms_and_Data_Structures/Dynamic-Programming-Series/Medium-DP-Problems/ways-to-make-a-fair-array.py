"""
Problem: Ways to Make a Fair Array
Difficulty: Medium

This problem gives you an array. You need to find the number of ways to remove exactly one element so that the sum of elements at even indices equals the sum at odd indices.

Approach:
Use prefix sums for even and odd indices. For each index, calculate the new even and odd sums if that element is removed. Count the number of indices where the sums are equal.
"""

def waysToMakeFair(nums):
    n = len(nums)  # Length of the array
    odd = [0]*(n+1)   # Prefix sum for odd indices
    even = [0]*(n+1)  # Prefix sum for even indices
    # Build prefix sums
    for i in range(n):
        odd[i+1] = odd[i]
        even[i+1] = even[i]
        if i % 2 == 0:
            even[i+1] += nums[i]  # Add to even sum
        else:
            odd[i+1] += nums[i]   # Add to odd sum
    res = 0  # Result to count the number of ways
    for i in range(n):
        # Calculate new even and odd sums if nums[i] is removed
        if even[i] + odd[n] - odd[i+1] == odd[i] + even[n] - even[i+1]:
            res += 1  # If sums are equal, increment result
    return res  # Return the total number of ways 