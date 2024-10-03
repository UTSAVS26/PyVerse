def knapSack(W, wt, val, n, memo=None):
    if memo is None:
        memo = {}
    
    if (n, W) in memo:
        return memo[(n, W)]

    if n == 0 or W == 0:
        return 0
    
    if wt[n-1] > W:
        memo[(n, W)] = knapSack(W, wt, val, n-1, memo)
    else:
        memo[(n, W)] = max(
            val[n-1] + knapSack(W - wt[n-1], wt, val, n-1, memo),
            knapSack(W, wt, val, n-1, memo)
        )
    
    return memo[(n, W)]

if __name__ == '__main__':
    profit = list(map(int, input("Enter profits separated by spaces: ").split()))
    weight = list(map(int, input("Enter weights separated by spaces: ").split()))
    W = int(input("Enter the capacity of the knapsack: "))
    n = len(profit)
    
    result = knapSack(W, weight, profit, n)
    print(f"The maximum value in the knapsack is: {result}")
