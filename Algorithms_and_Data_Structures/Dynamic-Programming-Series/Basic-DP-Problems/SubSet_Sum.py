# Problem: Given a set of integers, find if there is a subset with sum equal to a given number.


def subset_sum(nums, target):
    dp = [False] * (target + 1)
    dp[0] = True

    for num in nums:
        for i in range(target, num - 1, -1):
            dp[i] = dp[i] or dp[i - num]

    return dp[target]

# Example usage
nums = [3, 34, 4, 12, 5, 2]
target = 9
print(f"Is there a subset with sum {target}? {'Yes' if subset_sum(nums, target) else 'No'}")  # Output: Yes