import heapq

class PriorityQueue:
    def __init__(self):
        self.queue = []
        self.index = 0

    def is_empty(self):
        return len(self.queue) == 0

    def enqueue(self, item, priority):
        heapq.heappush(self.queue, (-priority, self.index, item))
        self.index += 1
        print(f"Enqueued: {item} with priority {priority}")

    def dequeue(self):
        if self.is_empty():
            print("Queue is empty. Cannot dequeue.")
            return None
        else:
            _, _, item = heapq.heappop(self.queue)
            print(f"Dequeued: {item}")
            return item

    def front(self):
        if self.is_empty():
            print("Queue is empty.")
            return None
        return self.queue[0][2]

    def display(self):
        print("Queue:", [(item, -priority) for priority, _, item in self.queue])

# Example usage
if __name__ == "__main__":
    pq = PriorityQueue()
    pq.enqueue("Task 1", 2)
    pq.enqueue("Task 2", 1)
    pq.enqueue("Task 3", 3)
    pq.display()
    pq.dequeue()
    pq.display()
    print("Front:", pq.front())