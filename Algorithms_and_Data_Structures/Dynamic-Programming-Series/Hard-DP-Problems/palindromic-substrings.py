def countPalindromicSubstrings(s):
    n = len(s)
    count = 0
    
    def expandAroundCenter(left, right):
        nonlocal count
        while left >= 0 and right < n and s[left] == s[right]:
            count += 1
            left -= 1
            right += 1
    
    # Consider each character and each gap between characters as a center
    for i in range(n):
        expandAroundCenter(i, i)       # Odd-length palindromes
        expandAroundCenter(i, i + 1)   # Even-length palindromes
    
    return count

# Test the function
s = "abc"
print(f"Number of palindromic substrings in '{s}': {countPalindromicSubstrings(s)}")

s = "aaa"
print(f"Number of palindromic substrings in '{s}': {countPalindromicSubstrings(s)}")
