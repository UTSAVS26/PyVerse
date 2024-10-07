def get_lcs_length(S1, S2):
    m = len(S1)  # Length of the first string
    n = len(S2)  # Length of the second string

    # Create a 2D list (dp table) to store the length of LCS for substrings
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Fill the dp table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # If characters match, add 1 to the previous diagonal value
            if S1[i - 1] == S2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                # If not, take the maximum value from the left or above cell
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # The last cell contains the length of the longest common subsequence
    return dp[m][n]

""" memoization method """
def lcs_memo(S1:str ,S2:str ,i:int ,j:int ,dp:list[list[int]]):
    """
    Compute the length of the longest common subsequence (LCS) between two strings using memoization.
    
    Parameters:
    S1 (str): The first string.
    S2 (str): The second string.
    i (int): Current index of string S1 (starts from len(S1)-1).
    j (int): Current index of string S2 (starts from len(S2)-1).
    dp (list[list[int]]): A memoization table initialized with -1 to store intermediate results.
    
    Returns:
    int: Length of the longest common subsequence between S1 and S2
    """
    #base case : if either string is ended return 0
    if i<0 or j<0:   
        return 0;
    
    #if result of subproblem is already calculated then no need to calculate again
    if dp[i][j] != -1: 
        return dp[i][j]
    
    if S1[i] == S2[j]:
        dp[i][j] = 1 + lcs_memo(S1,S2,i-1,j-1,dp)
    else:   
        dp[i][j] = max(lcs_memo(S1,S2,i-1,j,dp),lcs_memo(S1,S2,i,j-1,dp))
    
    return dp[i][j]

if __name__ == "__main__":
    # Take input strings from the user
    S1 = input("Enter the first string: ")
    S2 = input("Enter the second string: ")

    # Calculate the LCS length
    result = get_lcs_length(S1, S2)
    
    # Output the result
    print("Length of LCS is", result)
