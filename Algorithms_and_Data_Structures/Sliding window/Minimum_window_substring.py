class Solution:
    def minWindow(self, s: str, t: str) -> str:
        if not s or not t:
            return ""

        target_chars = [0] * 128  # ASCII characters
        window_chars = [0] * 128

        left = 0
        right = 0
        missing_chars = len(t)
        
        min_len = float('inf')
        min_window = ""

        # Count the frequency of each character in t
        for c in t:
            target_chars[ord(c)] += 1

        # Expand the window with the 'right' pointer
        while right < len(s):
            if target_chars[ord(s[right])] > 0:
                window_chars[ord(s[right])] += 1
                if window_chars[ord(s[right])] <= target_chars[ord(s[right])]:
                    missing_chars -= 1

            # When we have all the needed characters, start shrinking the window
            while missing_chars == 0:
                if right - left + 1 < min_len:
                    min_len = right - left + 1
                    min_window = s[left:right + 1]

                if target_chars[ord(s[left])] > 0:
                    window_chars[ord(s[left])] -= 1
                    if window_chars[ord(s[left])] < target_chars[ord(s[left])]:
                        missing_chars += 1

                left += 1

            right += 1

        return min_window
