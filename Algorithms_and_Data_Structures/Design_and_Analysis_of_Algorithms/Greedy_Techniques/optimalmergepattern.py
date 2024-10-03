class Heap:
    def __init__(self):
        self.h = []

    def parent(self, index):
        return (index - 1) // 2 if index > 0 else None

    def lchild(self, index):
        return (2 * index) + 1

    def rchild(self, index):
        return (2 * index) + 2

    def addItem(self, item):
        self.h.append(item)
        index = len(self.h) - 1
        parent = self.parent(index)

        while index > 0 and item < self.h[parent]:
            self.h[index], self.h[parent] = self.h[parent], self.h[index]
            index = parent
            parent = self.parent(index)

    def deleteItem(self):
        length = len(self.h)
        self.h[0], self.h[length - 1] = self.h[length - 1], self.h[0]
        deleted = self.h.pop()
        self.moveDownHeapify(0)
        return deleted

    def moveDownHeapify(self, index):
        lc, rc = self.lchild(index), self.rchild(index)
        length, smallest = len(self.h), index

        if lc < length and self.h[lc] < self.h[smallest]:
            smallest = lc
        if rc < length and self.h[rc] < self.h[smallest]:
            smallest = rc
        if smallest != index:
            self.h[smallest], self.h[index] = self.h[index], self.h[smallest]
            self.moveDownHeapify(smallest)

    def increaseItem(self, index, value):
        if value <= self.h[index]:
            return
        self.h[index] = value
        self.moveDownHeapify(index)


class OptimalMergePattern:
    def __init__(self, items):
        self.items = items
        self.heap = Heap()

    def optimalMerge(self):
        if len(self.items) <= 1:
            return sum(self.items)

        for item in self.items:
            self.heap.addItem(item)

        total_cost = 0
        while len(self.heap.h) > 1:
            first = self.heap.deleteItem()
            second = self.heap.h[0]
            total_cost += (first + second)
            self.heap.increaseItem(0, first + second)

        return total_cost


if __name__ == '__main__':
    n = int(input("Enter the number of items: "))
    items = list(map(int, input("Enter the item sizes separated by spaces: ").split()))
    omp = OptimalMergePattern(items)
    result = omp.optimalMerge()
    print("Optimal Merge Cost:", result)
