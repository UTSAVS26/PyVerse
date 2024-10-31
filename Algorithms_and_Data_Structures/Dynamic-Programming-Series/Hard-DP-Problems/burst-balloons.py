def maxCoins(nums):
    # Add virtual balloons with value 1 at each end to handle edge cases easily
    nums = [1] + nums + [1]
    n = len(nums)
    
    dp = [[0] * n for _ in range(n)]
    
    for length in range(2, n):  
        for left in range(0, n - length): 
            right = left + length  
            # Calculate max coins by bursting balloons in the range (left, right)
            for i in range(left + 1, right):
                dp[left][right] = max(dp[left][right], nums[left] * nums[i] * nums[right] + dp[left][i] + dp[i][right])

    # The answer is in dp[0][n-1], representing the entire range of the original array
    return dp[0][n - 1]

# Test the function
nums = [3, 1, 5, 8]
print(f"Maximum coins that can be collected: {maxCoins(nums)}")
