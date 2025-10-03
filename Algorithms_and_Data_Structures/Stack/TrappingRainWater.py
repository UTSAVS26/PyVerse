def trapping_rain_water(height):
    """
    LeetCode #42 - Trapping Rain Water
    ----------------------------------
    Given n non-negative integers representing an elevation map 
    where the width of each bar is 1, compute how much water is trapped.
    """
    n = len(height)
    if n == 0:
        return 0

    left_max = [0] * n
    right_max = [0] * n

    left_max[0] = height[0]
    for i in range(1, n):
        left_max[i] = max(height[i], left_max[i - 1])

    right_max[n - 1] = height[n - 1]
    for i in range(n - 2, -1, -1):
        right_max[i] = max(height[i], right_max[i + 1])

    total = 0
    for i in range(n):
        total += min(left_max[i], right_max[i]) - height[i]

    return total


if __name__ == "__main__":
    arr = list(map(int, input("Enter heights separated by space: ").split()))
    print("Water trapped:", trapping_rain_water(arr))
