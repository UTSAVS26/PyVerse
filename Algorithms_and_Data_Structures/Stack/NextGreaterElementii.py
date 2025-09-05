def next_greater_elements(nums):
    """
    LeetCode #503 - Next Greater Element II
    ---------------------------------------
    Given a circular array, find the next greater element 
    for each element in the array.
    """
    n = len(nums)
    ans = [-1] * n
    stack = []

    for i in range(2 * n - 1, -1, -1):
        while stack and stack[-1] <= nums[i % n]:
            stack.pop()
        if i < n:
            ans[i] = stack[-1] if stack else -1
        stack.append(nums[i % n])

    return ans


if __name__ == "__main__":
    arr = list(map(int, input("Enter numbers separated by space: ").split()))
    print("Next greater elements:", next_greater_elements(arr))
