def merge(arr, left, mid, right):
    # Merge two subarrays
    n1 = mid - left + 1  # Size of left subarray
    n2 = right - mid  # Size of right subarray

    L = [0] * n1  # Temp array for left
    R = [0] * n2  # Temp array for right

    # Copy data to temp arrays
    for i in range(n1):
        L[i] = arr[left + i]
    for j in range(n2):
        R[j] = arr[mid + 1 + j]

    i = j = 0  # Initial indexes for L[] and R[]
    k = left  # Initial index for merged array

    # Merge temp arrays back to arr
    while i < n1 and j < n2:
        if L[i] <= R[j]:
            arr[k] = L[i]  # Copy from L[]
            i += 1
        else:
            arr[k] = R[j]  # Copy from R[]
            j += 1
        k += 1

    # Copy remaining elements of L[]
    while i < n1:
        arr[k] = L[i]
        i += 1
        k += 1

    # Copy remaining elements of R[]
    while j < n2:
        arr[k] = R[j]
        j += 1
        k += 1

def merge_sort(arr, left, right):
    # Sort array using merge sort
    if left < right:  # Check if more than one element
        mid = (left + right) // 2  # Find mid point
        merge_sort(arr, left, mid)  # Sort first half
        merge_sort(arr, mid + 1, right)  # Sort second half
        merge(arr, left, mid, right)  # Merge halves

def getArrayInput():
    # Get array input from user
    while True:
        try:
            arr = list(map(int, input("Enter array elements (separated by spaces): ").split()))
            if not arr:
                print("Array cannot be empty.")
            else:
                return arr  # Return array
        except ValueError:
            print("Invalid input! Enter integers only.")

def print_list(arr):
    # Print array elements
    for i in arr:
        print(i, end=" ")
    print()  # New line

if __name__ == "__main__":
    arr = getArrayInput()  # Get user input
    print("Given array is:")
    print_list(arr)  # Print original array

    merge_sort(arr, 0, len(arr) - 1)  # Sort the array

    print("\nSorted array is:")
    print_list(arr)  # Print sorted array
