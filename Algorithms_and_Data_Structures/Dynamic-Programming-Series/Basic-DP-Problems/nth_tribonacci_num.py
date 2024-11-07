"""
Problem: Similar to the Fibonacci sequence, the Tribonacci sequence is defined as:
dp[n] = dp[n-1] + dp[n-2] + dp[n-3].
Given n, find the N-th Tribonacci number.
"""

def tribonacci(n):
    if n == 0:
        return 0
    elif n == 1 or n == 2:
        return 1

    dp = [0] * (n + 1)
    dp[0], dp[1], dp[2] = 0, 1, 1

    for i in range(3, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2] + dp[i - 3]

    return dp[n]

# Example usage
n = 10
print(f"The {n}-th Tribonacci number is: {tribonacci(n)}")  # Output: The 10-th Tribonacci number is: 149
