class Heap:
    def __init__(self):
        self.h = []  # Initialize the heap as an empty list

    def parent(self, index):
        # Return the index of the parent node
        return (index - 1) // 2 if index > 0 else None

    def lchild(self, index):
        # Return the index of the left child
        return (2 * index) + 1

    def rchild(self, index):
        # Return the index of the right child
        return (2 * index) + 2

    def addItem(self, item):
        # Add a new item to the heap
        self.h.append(item)  # Append item to the heap
        index = len(self.h) - 1  # Get the index of the newly added item
        parent = self.parent(index)  # Get the parent index

        # Move the new item up the heap to maintain heap property
        while index > 0 and item < self.h[parent]:
            self.h[index], self.h[parent] = self.h[parent], self.h[index]  # Swap
            index = parent  # Move up the heap
            parent = self.parent(index)

    def deleteItem(self):
        # Remove and return the minimum item (root) from the heap
        length = len(self.h)
        self.h[0], self.h[length - 1] = self.h[length - 1], self.h[0]  # Swap root with the last item
        deleted = self.h.pop()  # Remove the last item (the root)
        self.moveDownHeapify(0)  # Restore heap property
        return deleted

    def moveDownHeapify(self, index):
        # Move down the heap to restore heap property
        lc, rc = self.lchild(index), self.rchild(index)  # Get left and right children
        length, smallest = len(self.h), index  # Initialize smallest as the current index

        # Check if left child is smaller than the current smallest
        if lc < length and self.h[lc] < self.h[smallest]:
            smallest = lc
        # Check if right child is smaller than the current smallest
        if rc < length and self.h[rc] < self.h[smallest]:
            smallest = rc
        # If the smallest is not the current index, swap and continue heapifying
        if smallest != index:
            self.h[smallest], self.h[index] = self.h[index], self.h[smallest]
            self.moveDownHeapify(smallest)

    def increaseItem(self, index, value):
        # Increase the value of the item at the given index
        if value <= self.h[index]:
            return  # Do nothing if the new value is not greater
        self.h[index] = value  # Update the value
        self.moveDownHeapify(index)  # Restore heap property

class OptimalMergePattern:
    def __init__(self, items):
        self.items = items  # Store the items
        self.heap = Heap()  # Create a heap instance

    def optimalMerge(self):
        # Calculate the optimal merge cost using a min-heap
        if len(self.items) <= 1:
            return sum(self.items)  # If there's one or no item, return the sum

        # Add all items to the heap
        for item in self.items:
            self.heap.addItem(item)

        total_cost = 0  # Initialize total cost
        # Merge items until one item is left
        while len(self.heap.h) > 1:
            first = self.heap.deleteItem()  # Remove the smallest item
            second = self.heap.h[0]  # Get the next smallest item (root)
            total_cost += (first + second)  # Add their sum to the total cost
            self.heap.increaseItem(0, first + second)  # Merge them and add back to heap

        return total_cost  # Return the total merge cost

if __name__ == '__main__':
    n = int(input("Enter the number of items: "))  # Input the number of items
    items = list(map(int, input("Enter the item sizes separated by spaces: ").split()))  # Input item sizes
    omp = OptimalMergePattern(items)  # Create an instance of the OptimalMergePattern class
    result = omp.optimalMerge()  # Calculate optimal merge cost
    print("Optimal Merge Cost:", result)  # Output the result
