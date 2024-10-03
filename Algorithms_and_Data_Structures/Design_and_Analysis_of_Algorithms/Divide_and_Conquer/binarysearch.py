def binarySearch(arr, low, high, x):
    while low <= high:
        mid = low + (high - low) // 2

        if arr[mid] == x:
            return mid
        elif arr[mid] < x:
            low = mid + 1
        else:
            high = mid - 1

    return -1

def getArrayInput():
    while True:
        try:
            arr = list(map(int, input("Enter array elements (separated by spaces): ").split()))
            if not arr:
                print("Array cannot be empty. Please try again.")
            else:
                return arr
        except ValueError:
            print("Invalid input! Please enter integers only.")

def getElementInput():
    while True:
        try:
            x = int(input("Enter the element to search: "))
            return x
        except ValueError:
            print("Invalid input! Please enter an integer.")

if __name__ == '__main__':
    arr = getArrayInput()

    # Optionally sort the array if not sorted
    sort_choice = input("Do you want to sort the array before searching? (y/n): ").strip().lower()
    if sort_choice == 'y':
        arr.sort()
        print("Sorted array:", arr)
    
    x = getElementInput()

    result = binarySearch(arr, 0, len(arr) - 1, x)
    if result != -1:
        print(f"Element {x} is present at index {result}")
    else:
        print(f"Element {x} is not present in the array")
