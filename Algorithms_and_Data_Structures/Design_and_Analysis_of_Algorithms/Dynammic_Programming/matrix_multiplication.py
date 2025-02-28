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

""" memoization method """
def minMult_memo(arr:list ,i:int ,j:int ,dp:list[list[int]]):
    """
    Compute the minimum number of scalar multiplications required to multiply 
    a chain of matrices using memoization (Matrix Chain Multiplication problem).
    
    Parameters:
    arr (list): List of integers where the i-th matrix has dimensions arr[i-1] x arr[i].
    i (int): Starting index of the matrix chain.
    j (int): Ending index of the matrix chain.
    dp (list[list[int]]): A memoization table initialized with -1 to store intermediate results.
    
    Returns:
    int: Minimum number of scalar multiplications needed to multiply matrices from index i to j.
    if i==j:
        return 0
    """
    if dp[i][j] != -1:
        return dp[i][j]
    
    ans = sys.maxsize
    
    for k in range(i,j):
        
        res = minMult_memo(arr ,i ,k ,dp) + minMult_memo(arr ,k+1 ,j ,dp) + arr[i-1]* arr[j] * arr[k]
        
        ans = min(ans,res)
    
    dp[i][j] = ans
    return dp[i][j]

if __name__ == "__main__":
    # Input the matrix dimensions as space-separated integers
    arr = list(map(int, input("Enter the dimensions of matrices separated by spaces: ").split()))

    # Calculate the minimum number of scalar multiplications
    result = minMult(arr)

    # Output the result
    print(f"The minimum number of multiplications is: {result}")
