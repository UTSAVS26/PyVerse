def fibonacci(n):
    # Handle base cases
    if n == 0:
        return 0
    if n == 1:
        return 1

    # Create a table to store Fibonacci numbers
    fib = [0] * (n + 1)
    fib[1] = 1

    # Fill the table in a bottom-up manner
    for i in range(2, n + 1):
        fib[i] = fib[i - 1] + fib[i - 2]

    return fib[n]

# Test the function
n = 10
print(f"Fibonacci number at position {n} is: {fibonacci(n)}")
