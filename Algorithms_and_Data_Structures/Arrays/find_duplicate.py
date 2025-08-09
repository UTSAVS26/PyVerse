def input_array(arr, n):
    print(f"\nEnter {n} elements (range [1, {n-1}] with one duplicate):")
    for i in range(n):
        num = int(input(f"Enter element {i + 1}: "))
        arr.append(num)

def display_array(arr):
    print("\nInput array is:")
    for x in arr:
        print(f"{x}\t", end="")
    print()

def find_duplicate(nums):
    # Phase 1: Detect cycle
    slow = nums[0]
    fast = nums[0]

    while True:
        slow = nums[slow]
        fast = nums[nums[fast]]
        if slow == fast:
            break

    # Phase 2: Find entrance to the cycle (duplicate number)
    slow = nums[0]
    while slow != fast:
        slow = nums[slow]
        fast = nums[fast]

    return slow

if __name__ == "__main__":
    nums = []
    n = int(input("Enter the number of elements (n + 1): "))
    input_array(nums, n)
    display_array(nums)

    duplicate = find_duplicate(nums)
    print(f"\nThe duplicate number is: {duplicate}")

