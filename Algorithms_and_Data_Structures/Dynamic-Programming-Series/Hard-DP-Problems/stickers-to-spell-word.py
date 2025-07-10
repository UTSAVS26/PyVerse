"""
Stickers to Spell Word
Given a list of stickers (each sticker is a string) and a target word, find the minimum number of stickers required to spell out the target word. Each sticker can be used multiple times. If it's not possible, return -1.
"""

from collections import Counter

class Solution:
    def minStickers(self, stickers, target):
        # Preprocess stickers to count letters
        sticker_counts = [Counter(sticker) for sticker in stickers]
        memo = {}
        
        # Helper function to solve for the remaining target
        def dp(remain):
            if remain == '':
                return 0  # No letters left, no stickers needed
            if remain in memo:
                return memo[remain]  # Already solved this state
            target_count = Counter(remain)
            res = float('inf')
            for sticker in sticker_counts:
                # If sticker doesn't help with the first letter, skip
                if remain[0] not in sticker:
                    continue
                # Build new remaining string after using this sticker
                new_remain = ''
                for c in target_count:
                    if target_count[c] > sticker.get(c, 0):
                        new_remain += c * (target_count[c] - sticker.get(c, 0))
                # Try using this sticker and solve for the rest
                sub_res = dp(new_remain)
                if sub_res != -1:
                    res = min(res, 1 + sub_res)
            memo[remain] = -1 if res == float('inf') else res
            return memo[remain]
        
        return dp(target) 