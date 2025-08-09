def input_array(arr, n):
    for i in range(n):
        num = int(input(f"Enter element {i + 1}: "))
        arr.append(num)

def display_array(arr):
    print("\nInput array is:")
    for x in arr:
        print(f"{x}\t", end="")
    print()

def find_peak(nums, n):
    left = 0
    right = n - 1

    while left < right:
        mid = (left + right) // 2

        # Check slope direction to decide the half
        if nums[mid] > nums[mid + 1]:
            # Mid might be the peak or peak lies on the left side
            right = mid
        else:
            # Peak lies to the right of mid
            left = mid + 1

    # At the end of loop, left == right and pointing to a peak
    return left

if __name__ == "__main__":
    nums = []
    n = int(input("Enter number of elements in array: "))
    input_array(nums, n)
    display_array(nums)

    peak_index = find_peak(nums, n)
    print(f"\nPeak element is at index {peak_index} with value {nums[peak_index]}")

