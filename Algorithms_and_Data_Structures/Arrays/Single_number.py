def find_single_element(nums, n):
    # Variables to hold bits seen once and twice
    once = 0
    twice = 0

    for i in range(n):
        num = nums[i]
        # 'twice' holds bits which appear twice
        twice |= once & num
        # 'once' holds bits which appear once
        once ^= num
        # common_mask removes bits appearing three times
        common_mask = ~(once & twice)
        once &= common_mask
        twice &= common_mask

    return once
    
def input_array(arr, n):
    for i in range(n):
       num = int(input(f"Enter element {i + 1}: "))
       arr.append(num)

def display_array(arr):
    print("\nInput array is:")
    for x in arr:
        print(f"{x}\t", end="")
    print()

if __name__ == "__main__":
    nums = []
    n = int(input("Enter number of elements in array: "))
    input_array(nums, n)
    display_array(nums)

    result = find_single_element(nums, n)
    print(f"\nThe element that appears only once is: {result}")

