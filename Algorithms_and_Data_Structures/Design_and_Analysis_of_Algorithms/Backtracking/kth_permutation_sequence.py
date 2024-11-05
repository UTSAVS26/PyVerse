class Solution:
    def getPermutation(self, n: int, k: int) -> str:
        # Initialize the list of numbers from 1 to n
        numbers = list(range(1, n + 1))
        # Initialize the result string
        result = []
        # Decrement k by 1 to convert to 0-based index
        k -= 1
        
        def backtrack(nums, k, path):
            if not nums:
                # If no numbers left, we've found the kth permutation
                return True
            
            # Calculate the factorial of the remaining numbers
            factorial = 1
            for i in range(1, len(nums)):
                factorial *= i
            
            for i in range(len(nums)):
                if k >= factorial:
                    # Skip this number, as it's not part of the kth permutation
                    k -= factorial
                else:
                    # Add the current number to the path
                    path.append(str(nums[i]))
                    # Remove the used number from the list
                    nums.pop(i)
                    # Recursively build the rest of the permutation
                    if backtrack(nums, k, path):
                        return True
                    # If not successful, backtrack
                    nums.insert(i, int(path.pop()))
                
                # Move to the next number
                k %= factorial
            
            return False
        
        # Start the backtracking process
        backtrack(numbers, k, result)
        
        # Join the result list into a string and return
        return ''.join(result)

# Example usage
if __name__ == "__main__":
    solution = Solution()
    n = 4
    k = 9
    print(f"The {k}th permutation of {n} numbers is: {solution.getPermutation(n, k)}")