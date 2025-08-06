# Python Program to reverse an array using Two Pointers

# Function to reverse an array
def reverseArray(arr):
    left = 0
    right = len(arr) - 1

    while left < right:
        # Swap elements
        arr[left], arr[right] = arr[right], arr[left]
        left += 1
        right -= 1

if __name__ == "__main__":
    # Taking user input for array elements
    arr = list(map(int, input("Enter array elements separated by space: ").split()))

    reverseArray(arr)

    print("Reversed array:")
    print(*arr)

