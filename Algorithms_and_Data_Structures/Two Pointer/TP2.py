# Max longest ones |||

class TP2:
    
    def longestOnes(self, nums: List[int], k: int) -> int:
        l=r=0    
        for r in range(len(nums)):
            if nums[r] == 0:
                k-=1
            if k<0:
                if nums[l] == 0:
                    k+=1
                l+=1
        return r-l+1

tp2 = TP2()


nums = [1,1,1,0,0,0,1,1,1,1,0]
k = 2

result = tp2.longestOnes(nums, k)
print("The length of the longest subarray with at most k zeros is:", result)