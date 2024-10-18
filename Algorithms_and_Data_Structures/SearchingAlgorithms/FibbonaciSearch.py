def fibonacci_search(arr, target):
    fib_m_2 = 0
    fib_m_1 = 1
    fib_m = fib_m_1 + fib_m_2
    
    while fib_m < len(arr):
        fib_m_2 = fib_m_1
        fib_m_1 = fib_m
        fib_m = fib_m_1 + fib_m_2

    offset = -1

    while fib_m > 1:
        i = min(offset + fib_m_2, len(arr) - 1)

        if arr[i] < target:
            fib_m = fib_m_1
            fib_m_1 = fib_m_2
            fib_m_2 = fib_m - fib_m_1
            offset = i
        elif arr[i] > target:
            fib_m = fib_m_2
            fib_m_1 = fib_m_1 - fib_m_2
            fib_m_2 = fib_m - fib_m_1
        else:
            return i

    if fib_m_1 and arr[offset + 1] == target:
        return offset + 1

    return -1

# Example usage
arr = [1, 2, 3, 4, 5, 6]
target = 4
print(fibonacci_search(arr, target))
