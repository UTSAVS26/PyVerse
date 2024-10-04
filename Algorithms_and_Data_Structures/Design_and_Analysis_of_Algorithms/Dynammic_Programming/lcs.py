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

if __name__ == "__main__":
    # Take input strings from the user
    S1 = input("Enter the first string: ")
    S2 = input("Enter the second string: ")

    # Calculate the LCS length
    result = get_lcs_length(S1, S2)
    
    # Output the result
    print("Length of LCS is", result)
