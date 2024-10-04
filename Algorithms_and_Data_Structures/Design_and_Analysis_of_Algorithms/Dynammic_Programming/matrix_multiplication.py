import sys

def minMult(arr):
    n = len(arr)  # Length of the array

    # Initialize a 2D table dp to store the minimum cost of matrix multiplication
    dp = [[0] * n for _ in range(n)]

    # l is the chain length (number of matrices involved)
    for l in range(2, n):  # Start from chain length 2 since l=1 is trivial
        for i in range(n - l):
            j = i + l  # End of the chain
            dp[i][j] = sys.maxsize  # Initialize with a large value

            # Try placing parentheses at different positions
            for k in range(i + 1, j):
                # Calculate the cost of multiplying matrices from i to j with split at k
                q = dp[i][k] + dp[k][j] + arr[i] * arr[k] * arr[j]
                # Store the minimum cost in dp[i][j]
                dp[i][j] = min(dp[i][j], q)

    # Return the minimum cost to multiply the entire chain of matrices
    return dp[0][n - 1]

if __name__ == "__main__":
    # Input the matrix dimensions as space-separated integers
    arr = list(map(int, input("Enter the dimensions of matrices separated by spaces: ").split()))

    # Calculate the minimum number of scalar multiplications
    result = minMult(arr)

    # Output the result
    print(f"The minimum number of multiplications is: {result}")
