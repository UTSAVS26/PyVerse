class Solution:
    def subarraysWithKDistinct(self, nums: List[int], k: int) -> int:
        return self.slidingWindowAtMost(nums, k) - self.slidingWindowAtMost(nums, k - 1)
    
    def slidingWindowAtMost(self, nums: List[int], distinct_k: int) -> int:
        freq_map = {}
        left = 0
        total_count = 0

        for right in range(len(nums)):
            # Add current number to the frequency map
            if nums[right] in freq_map:
                freq_map[nums[right]] += 1
            else:
                freq_map[nums[right]] = 1

            # Shrink the window until we have at most 'distinct_k' distinct numbers
            while len(freq_map) > distinct_k:
                freq_map[nums[left]] -= 1
                if freq_map[nums[left]] == 0:
                    del freq_map[nums[left]]
                left += 1

            # Add the number of subarrays ending at 'right'
            total_count += (right - left + 1)

        return total_count
