import sys

def minMult(arr):
    n = len(arr)
    dp = [[0] * n for _ in range(n)]

    for l in range(2, n):
        for i in range(n - l):
            j = i + l
            dp[i][j] = sys.maxsize
            for k in range(i + 1, j):
                q = dp[i][k] + dp[k][j] + arr[i] * arr[k] * arr[j]
                dp[i][j] = min(dp[i][j], q)

    return dp[0][n - 1]

if __name__ == "__main__":
    arr = list(map(int, input("Enter the dimensions of matrices separated by spaces: ").split()))
    result = minMult(arr)
    print(f"The minimum number of multiplications is: {result}")
