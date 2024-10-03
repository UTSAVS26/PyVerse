def get_lcs_length(S1, S2):
    m = len(S1)
    n = len(S2)

    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if S1[i - 1] == S2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]

if __name__ == "__main__":
    S1 = input("Enter the first string: ")
    S2 = input("Enter the second string: ")
    result = get_lcs_length(S1, S2)
    print("Length of LCS is", result)
