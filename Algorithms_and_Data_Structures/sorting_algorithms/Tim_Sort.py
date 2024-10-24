
# Define the minimum run size
RUN = 32

# Function - perform insertion sort on subarray
def insertion_sort(arr, left, right):
    for i in range(left + 1, right + 1):
        temp = arr[i]
        j = i - 1
        while j >= left and arr[j] > temp:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = temp

# Function to merge two sorted subarrays
def merge(arr, l, m, r):
    # Create temporary arrays to hold the two halves
    len1, len2 = m - l + 1, r - m
    left = arr[l:l + len1]
    right = arr[m + 1:m + 1 + len2]

    i, j, k = 0, 0, l

    # Merge the left and right arrays
    while i < len1 and j < len2:
        if left[i] <= right[j]:
            arr[k] = left[i]
            i += 1
        else:
            arr[k] = right[j]
            j += 1
        k += 1

    # Copy remaining elements from the left array
    while i < len1:
        arr[k] = left[i]
        i += 1
        k += 1

    # Copy remaining elements from the right array
    while j < len2:
        arr[k] = right[j]
        j += 1
        k += 1

# Function to perform TimSort
def tim_sort(arr):
    n = len(arr)

    # Sort individual subarrays of size RUN
    for i in range(0, n, RUN):
        insertion_sort(arr, i, min((i + RUN - 1), (n - 1)))

    # Start merging from size RUN. Merge subarrays in bottom-up manner
    size = RUN
    while size < n:
        for left in range(0, n, 2 * size):
            mid = left + size - 1
            right = min((left + 2 * size - 1), (n - 1))
            if mid < right:
                merge(arr, left, mid, right)
        size *= 2

# Function to print the array
def print_array(arr):
    for i in range(len(arr)):
        print(arr[i], end=" ")
    print()

# Driver code
if __name__ == "__main__":
    arr = [40, 12, 31, 27, 25, 8, 1, 32, 17]
    print("Before sorting array elements are - ")
    print_array(arr)

    tim_sort(arr)

    print("\nAfter sorting array elements are - ")
    print_array(arr)
