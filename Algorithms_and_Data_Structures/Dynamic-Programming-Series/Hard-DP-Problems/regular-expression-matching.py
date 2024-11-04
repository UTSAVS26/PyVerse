def isMatch(s, p):
    m, n = len(s), len(p)
    
    # Initialize a (m+1) x (n+1) DP table
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    
    dp[0][0] = True
    
    for j in range(1, n + 1):
        if p[j - 1] == '*':
            dp[0][j] = dp[0][j - 2]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j - 1] == s[i - 1] or p[j - 1] == '.':
                dp[i][j] = dp[i - 1][j - 1]
            elif p[j - 1] == '*':
                dp[i][j] = dp[i][j - 2] or (dp[i - 1][j] and (p[j - 2] == s[i - 1] or p[j - 2] == '.'))
    
    return dp[m][n]

s = "aab"
p = "c*a*b"
print(f"Does '{s}' match the pattern '{p}': {isMatch(s, p)}")
