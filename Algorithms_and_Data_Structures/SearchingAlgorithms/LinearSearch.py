def linear_search(arr, target):
    for index, value in enumerate(arr):
        if value == target:
            return index
    return -1

# Example usage
arr = [1, 2, 3, 4, 5, 6]
target = 4
print(linear_search(arr, target))
