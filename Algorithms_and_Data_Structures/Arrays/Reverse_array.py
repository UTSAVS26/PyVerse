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
    try:
        user_input = input("Enter array elements separated by space: ").strip()
        if not user_input:
            print("No elements entered. Using empty array.")
            arr = []
        else:
            arr = list(map(int, user_input.split()))
    except ValueError:
        print("Error: Please enter valid integers separated by spaces.")
        exit(1)

    reverseArray(arr)

    print("Reversed array:")
    print(*arr)

