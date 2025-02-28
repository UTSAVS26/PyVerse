def nth_fibonacci(n, memo={}):
    # Check if Fibonacci value is already calculated and stored in memo
    if n in memo:
        return memo[n]
    
    # Base cases: return n if n is 0 or 1 (Fibonacci(0) = 0, Fibonacci(1) = 1)
    if n <= 1:
        return n
    
    # Recursive case: calculate and store Fibonacci(n) in memo
    memo[n] = nth_fibonacci(n - 1, memo) + nth_fibonacci(n - 2, memo)
    return memo[n]

""" tabulation method for solving nth_fibonnaci """
def nth_fibonacci_tab(n:int) -> int:
    """ 
    Calculate n_th fibonnaci number in O(n)
    params:
    n : term value
    
    returns:
    int: nth term of fibonnaci sequence
    """
    dp = [0] * (n+1)
    dp[1] = 1
    for i in range(2,n+1):
        
        dp[i] = dp[i-1] + dp[i-2]
    
    return dp[n]

# Get input from the user for the Fibonacci number position
n = int(input("Enter the position of the Fibonacci number to find: "))

# Calculate the nth Fibonacci number
result = nth_fibonacci_tab(n)

# Print the result
print(f"The {n}th Fibonacci number is: {result}")
