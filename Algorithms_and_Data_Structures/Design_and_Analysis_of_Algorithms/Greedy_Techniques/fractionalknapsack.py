class Item:
    def __init__(self, profit, weight):
        self.profit = profit
        self.weight = weight

def fractionalKnapsack(W, arr):
    arr.sort(key=lambda x: (x.profit / x.weight), reverse=True)
    finalvalue = 0.0

    for item in arr:
        if item.weight <= W:
            W -= item.weight
            finalvalue += item.profit
        else:
            finalvalue += item.profit * W / item.weight
            break
    
    return finalvalue

if __name__ == "__main__":
    W = float(input("Enter the capacity of the knapsack: "))
    n = int(input("Enter the number of items: "))
    arr = []

    for _ in range(n):
        profit = float(input("Enter profit of item: "))
        weight = float(input("Enter weight of item: "))
        arr.append(Item(profit, weight))

    max_val = fractionalKnapsack(W, arr)
    print("Maximum value in the knapsack:", max_val)
