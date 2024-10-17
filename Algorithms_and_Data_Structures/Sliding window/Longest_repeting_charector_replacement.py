class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
        count = [0] * 26  # To store the frequency of each character
        max_count = 0      # To store the frequency of the most frequent character in the current window
        left = 0           # Left pointer of the sliding window
        result = 0         # To store the result
        
        # Iterate through the string with the 'right' pointer
        for right in range(len(s)):
            count[ord(s[right]) - ord('A')] += 1  # Increment the frequency of the current character
            
            # Update the max_count of the most frequent character in the window
            max_count = max(max_count, count[ord(s[right]) - ord('A')])
            
            # If the size of the window minus the max frequency character is greater than k, shrink the window
            if (right - left + 1) - max_count > k:
                count[ord(s[left]) - ord('A')] -= 1  # Decrease the frequency of the character being left out
                left += 1  # Move the left pointer to shrink the window
            
            result = max(result, right - left + 1)  # Update the result with the maximum window size
        
        return result
