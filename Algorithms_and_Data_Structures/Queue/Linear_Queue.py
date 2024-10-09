class LinearQueue:
    def __init__(self, max_size):
        self.queue = []
        self.max_size = max_size

    def is_empty(self):
        return len(self.queue) == 0

    def is_full(self):
        return len(self.queue) == self.max_size

    def enqueue(self, item):
        if self.is_full():
            print("Queue is full. Cannot enqueue.")
        else:
            self.queue.append(item)
            print(f"Enqueued: {item}")

    def dequeue(self):
        if self.is_empty():
            print("Queue is empty. Cannot dequeue.")
            return None
        else:
            item = self.queue.pop(0)
            print(f"Dequeued: {item}")
            return item

    def front(self):
        if self.is_empty():
            print("Queue is empty.")
            return None
        return self.queue[0]

    def display(self):
        print("Queue:", self.queue)

# Example usage
if __name__ == "__main__":
    q = LinearQueue(5)
    q.enqueue(1)
    q.enqueue(2)
    q.enqueue(3)
    q.display()
    q.dequeue()
    q.display()
    print("Front:", q.front())