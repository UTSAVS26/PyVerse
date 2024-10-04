def max_min_naive(arr):
    # Initialize max_val and min_val with the first element of the array
    max_val = arr[0]
    min_val = arr[0]
    
    # Iterate through the array starting from the second element
    for i in range(1, len(arr)):
        # Update max_val if the current element is greater
        if arr[i] > max_val:
            max_val = arr[i]
        # Update min_val if the current element is smaller
        if arr[i] < min_val:
            min_val = arr[i]
    
    # Return the maximum and minimum values
    return max_val, min_val

def getArrayInput():
    # Continuously prompt the user for input until valid data is provided
    while True:
        try:
            # Split the input string into integers and store them in an array
            arr = list(map(int, input("Enter array elements (separated by spaces): ").split()))
            if not arr:
                print("Array cannot be empty. Please try again.")
            else:
                return arr  # Return the valid array
        except ValueError:
            print("Invalid input! Please enter integers only.")

if __name__ == '__main__':
    # Get array input from the user
    arr = getArrayInput()
    
    # Find the maximum and minimum values using max_min_naive function
    max_val, min_val = max_min_naive(arr)
    
    # Print the maximum and minimum values
    print("Maximum element is:", max_val)
    print("Minimum element is:", min_val)
