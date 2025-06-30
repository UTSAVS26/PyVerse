def shellSort(arr):
    n = len(arr)
    gap = n // 2
    while gap > 0:
        for i in range(gap, n):
            temp = arr[i]
            j = i
            while j >= gap and arr[j - gap] > temp:
                arr[j] = arr[j - gap]
                j -= gap
            arr[j] = temp
        gap //= 2

# Example usage
data = [9, 8, 3, 7, 5, 6, 4, 1]
shellSort(data)
print("Sorted Array in Ascending Order:")
print(data)
