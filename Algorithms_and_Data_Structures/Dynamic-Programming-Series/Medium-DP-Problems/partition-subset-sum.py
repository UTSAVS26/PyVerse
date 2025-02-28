def canPartition(nums):
    total_sum = sum(nums)
    
    # If the total sum is odd, we can't partition it into two equal subsets
    if total_sum % 2 != 0:
        return False
    
    target = total_sum // 2
    n = len(nums)
    
    # DP array to store whether a specific sum is possible
    dp = [False] * (target + 1)
    dp[0] = True
    
    for num in nums:
        for i in range(target, num - 1, -1):
            dp[i] = dp[i] or dp[i - num]
    
    # If dp[target] is True, then it's possible to partition the array
    return dp[target]

nums = [1, 5, 11, 5]
print(f"Can partition: {canPartition(nums)}")
