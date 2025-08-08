def input_array(nums, n):
    for i in range(n):
        num = int(input(f"Enter element {i + 1}: "))
        nums.append(num)

def display_array(nums, k):
    print(f"\nFinal array with at most two duplicates (length {k}):")
    for i in range(k):
        print(f"{nums[i]}\t", end="")
    print()

def remove_duplicates(nums, n):
    # If there are 2 or fewer elements, no need to remove anything
    if n <= 2:
        return n

    insert_pos = 2  # First two elements are always allowed

    for i in range(2, n):
        # Only allow if current number is not same as the one two places before
        if nums[i] != nums[insert_pos - 2]:
            nums[insert_pos] = nums[i]
            insert_pos += 1

    return insert_pos  # New length of the array

if __name__ == "__main__":
    nums = []
    n = int(input("Enter number of elements in sorted array: "))
    input_array(nums, n)

    print("\nOriginal array:")
    for x in nums:
        print(f"{x}\t", end="")
    print()

    k = remove_duplicates(nums, n)
    display_array(nums, k)
