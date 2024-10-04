def partition(arr, low, high):
    # Set the last element as the pivot
    pivot = arr[high]
    # Initialize i to be one position before the low index
    i = low - 1
    
    # Traverse the array from low to high-1
    for j in range(low, high):
        # If the current element is smaller than the pivot
        if arr[j] < pivot:
            i += 1  # Increment the index for the smaller element
            swap(arr, i, j)  # Swap the current element with the element at index i
    
    # Swap the pivot with the element at i+1 to place it in the correct position
    swap(arr, i + 1, high)
    return i + 1  # Return the partition index

def swap(arr, i, j):
    # Swap the elements at index i and j
    arr[i], arr[j] = arr[j], arr[i]

def quickSort(arr, low, high):
    # Perform QuickSort only if the low index is less than the high index
    if low < high:
        # Get the partition index
        pi = partition(arr, low, high)
        # Recursively sort elements before and after the partition index
        quickSort(arr, low, pi - 1)  # Sort elements on the left of pivot
        quickSort(arr, pi + 1, high)  # Sort elements on the right of pivot

if __name__ == "__main__":
    # Take input from the user as a list of integers
    arr = list(map(int, input("Enter the numbers to sort, separated by spaces: ").split()))
    # Call QuickSort on the entire array
    quickSort(arr, 0, len(arr) - 1)
    # Print the sorted array
    print("Sorted array:", *arr)
