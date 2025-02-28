def ternary_search(arr, target, left, right):
    if left > right:
        return -1

    third_length = (right - left) // 3
    mid1 = left + third_length
    mid2 = right - third_length

    if arr[mid1] == target:
        return mid1
    elif arr[mid2] == target:
        return mid2
    elif target < arr[mid1]:
        return ternary_search(arr, target, left, mid1 - 1)
    elif target > arr[mid2]:
        return ternary_search(arr, target, mid2 + 1, right)
    else:
        return ternary_search(arr, target, mid1 + 1, mid2 - 1)

# Example usage
arr = [1, 2, 3, 4, 5, 6]
target = 4
print(ternary_search(arr, target, 0, len(arr) - 1))
