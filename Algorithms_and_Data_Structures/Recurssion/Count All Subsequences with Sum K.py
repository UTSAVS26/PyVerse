def countSubsequences(arr, k):
    def count(i, total):
        if i == len(arr):
            return 1 if total == k else 0
        
        # Include current element
        include = count(i + 1, total + arr[i])
        # Exclude current element
        exclude = count(i + 1, total)
        return include + exclude
    
    return count(0, 0)

# Example
arr = [1, 2, 1]
k = 2
print("Count of subsequences with sum", k, ":", countSubsequences(arr, k))
