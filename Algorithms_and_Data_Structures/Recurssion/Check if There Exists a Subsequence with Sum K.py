def existsSubsequence(arr, k):
    def exists(i, total):
        if i == len(arr):
            return total == k
        # Return true if any path returns true
        return exists(i + 1, total + arr[i]) or exists(i + 1, total)
    
    return exists(0, 0)

# Example
arr = [1, 2, 1]
k = 3
print("Exists subsequence with sum", k, ":", existsSubsequence(arr, k))
