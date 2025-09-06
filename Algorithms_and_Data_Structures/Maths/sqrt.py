 def input_value() -> int:
    x = int(input("Enter non-negative integer x: "))
    return x

def sqrt_floor(x: int) -> int:
    if x < 0:
        raise ValueError("x must be non-negative")
    if x < 2:
        return x  # sqrt(0)=0, sqrt(1)=1

    left, right = 1, x
    ans = 1
    while left <= right:
        mid = (left + right) // 2
        if mid * mid == x:
            return mid
        elif mid * mid < x:
            ans = mid
            left = mid + 1
        else:
            right = mid - 1
    return ans

if __name__ == "__main__":
    x = input_value()
    result = sqrt_floor(x)
    print(f"Square root of {x} (rounded down) is: {result}\n")
