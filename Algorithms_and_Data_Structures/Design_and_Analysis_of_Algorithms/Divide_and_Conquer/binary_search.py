def binarySearch(arr, low, high, x):
    """
    Performs binary search on a sorted array to find the index of element x.

    Parameters:
    arr (list): The sorted array to search.
    low (int): The starting index of the array segment.
    high (int): The ending index of the array segment.
    x (int): The element to search for.

    Returns:
    int: The index of element x in arr, or -1 if not found.
    """
    while low <= high:
        mid = low + (high - low) // 2  # Calculate mid index

        if arr[mid] == x:
            return mid  # Element found
        elif arr[mid] < x:
            low = mid + 1  # Search in the right half
        else:
            high = mid - 1  # Search in the left half

    return -1  # Element not found

def getArrayInput():
    """
    Prompts the user to input an array of integers.

    Returns:
    list: A list of integers input by the user.
    """
    while True:
        try:
            arr = list(map(int, input("Enter array elements (separated by spaces): ").split()))
            if not arr:
                print("Array cannot be empty. Please try again.")
            else:
                return arr  # Return the input array
        except ValueError:
            print("Invalid input! Please enter integers only.")

def getElementInput():
    """
    Prompts the user to input an integer to search in the array.

    Returns:
    int: The integer input by the user.
    """
    while True:
        try:
            x = int(input("Enter the element to search: "))
            return x  # Return the input element
        except ValueError:
            print("Invalid input! Please enter an integer.")

if __name__ == '__main__':
    # Get user input for the array
    arr = getArrayInput()

    # Optionally sort the array if not sorted
    sort_choice = input("Do you want to sort the array before searching? (y/n): ").strip().lower()
    if sort_choice == 'y':
        arr.sort()  # Sort the array
        print("Sorted array:", arr)
    
    # Get user input for the element to search
    x = getElementInput()

    # Perform binary search
    result = binarySearch(arr, 0, len(arr) - 1, x)
    if result != -1:
        print(f"Element {x} is present at index {result}")
    else:
        print(f"Element {x} is not present in the array")
