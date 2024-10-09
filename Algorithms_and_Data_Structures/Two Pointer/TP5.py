
# binary-subarrays-with-sum

class TP5(object):
    def numSubarraysWithSum(self, nums, goal):
        count = {0: 1}
        curr_sum = 0
        total_subarrays = 0
        
        for num in nums:
            curr_sum += num
            if curr_sum - goal in count:
                total_subarrays += count[curr_sum - goal]
            count[curr_sum] = count.get(curr_sum, 0) + 1

        return total_subarrays

tp5 = TP5()

nums = [10, 2, -2, 20, 10]
goal = 0

result = tp5.numSubarraysWithSum(nums, goal)
print("The number of subarrays with sum", goal, "is:", result)