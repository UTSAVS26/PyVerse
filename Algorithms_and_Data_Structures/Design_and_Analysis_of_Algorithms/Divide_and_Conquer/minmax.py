def max_min_naive(arr):
    max_val = arr[0]
    min_val = arr[0]
    for i in range(1, len(arr)):
        if arr[i] > max_val:
            max_val = arr[i]
        if arr[i] < min_val:
            min_val = arr[i]
    return max_val, min_val

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

if __name__ == '__main__':
    arr = getArrayInput()
    max_val, min_val = max_min_naive(arr)
    print("Maximum element is:", max_val)
    print("Minimum element is:", min_val)
