def nth_fibonacci(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = nth_fibonacci(n - 1, memo) + nth_fibonacci(n - 2, memo)
    return memo[n]

n = int(input("Enter the position of the Fibonacci number to find: "))
result = nth_fibonacci(n)
print(f"The {n}th Fibonacci number is: {result}")
