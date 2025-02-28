def knapSack(W, wt, val, n, memo=None):
    # Initialize memo dictionary if it's not provided
    if memo is None:
        memo = {}
    
    # Check if the solution for (n, W) is already memoized
    if (n, W) in memo:
        return memo[(n, W)]

    # Base case: No items left or knapsack capacity is 0
    if n == 0 or W == 0:
        return 0
    
    # If the weight of the nth item is more than the remaining capacity, skip it
    if wt[n-1] > W:
        memo[(n, W)] = knapSack(W, wt, val, n-1, memo)
    else:
        # Find the maximum of including or excluding the current item
        memo[(n, W)] = max(
            val[n-1] + knapSack(W - wt[n-1], wt, val, n-1, memo),  # Include the item
            knapSack(W, wt, val, n-1, memo)  # Exclude the item
        )
    
    # Return the memoized value
    return memo[(n, W)]

if __name__ == '__main__':
    # Input profits (values of items)
    profit = list(map(int, input("Enter profits separated by spaces: ").split()))
    
    # Input weights of the items
    weight = list(map(int, input("Enter weights separated by spaces: ").split()))
    
    # Input the total capacity of the knapsack
    W = int(input("Enter the capacity of the knapsack: "))
    
    # Number of items
    n = len(profit)
    
    # Get the maximum value for the knapsack
    result = knapSack(W, weight, profit, n)
    
    # Output the result
    print(f"The maximum value in the knapsack is: {result}")
