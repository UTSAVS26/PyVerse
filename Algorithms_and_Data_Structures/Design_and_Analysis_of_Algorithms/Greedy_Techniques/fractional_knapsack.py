class Item:
    def __init__(self, profit, weight):
        self.profit = profit  # Initialize profit of the item
        self.weight = weight  # Initialize weight of the item

def fractionalKnapsack(W, arr):
    # Sort the items based on their profit-to-weight ratio in descending order
    arr.sort(key=lambda x: (x.profit / x.weight), reverse=True)
    finalvalue = 0.0  # Initialize the final value of the knapsack

    for item in arr:
        # If the item can fully fit in the knapsack
        if item.weight <= W:
            W -= item.weight  # Reduce the remaining capacity
            finalvalue += item.profit  # Add the full profit of the item
        else:
            # If the item can't fully fit, take the fractional part
            finalvalue += item.profit * W / item.weight
            break  # No more capacity left, break the loop
    
    return finalvalue  # Return the maximum value

if __name__ == "__main__":
    W = float(input("Enter the capacity of the knapsack: "))  # Input the knapsack's capacity
    n = int(input("Enter the number of items: "))  # Input the number of items
    arr = []

    for _ in range(n):
        profit = float(input("Enter profit of item: "))  # Input profit for each item
        weight = float(input("Enter weight of item: "))  # Input weight for each item
        arr.append(Item(profit, weight))  # Create an item and add it to the list

    max_val = fractionalKnapsack(W, arr)  # Calculate the maximum value using the fractional knapsack algorithm
    print("Maximum value in the knapsack:", max_val)  # Output the result
