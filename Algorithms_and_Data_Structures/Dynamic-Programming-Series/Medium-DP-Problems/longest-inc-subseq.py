def lengthOfLIS(nums):
    if not nums:
        return 0

    # Create a dp array to store the length of the longest increasing subsequence up to each index
    dp = [1] * len(nums)  

    for i in range(1, len(nums)):
        for j in range(i):
            if nums[i] > nums[j]:  # If nums[i] can extend the subsequence ending at nums[j]
                dp[i] = max(dp[i], dp[j] + 1)

    return max(dp)

nums = [10, 9, 2, 5, 3, 7, 101, 18]
print(f"Length of the longest increasing subsequence: {lengthOfLIS(nums)}")
