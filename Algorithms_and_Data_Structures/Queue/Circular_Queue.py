class CircularQueue:
    def __init__(self, max_size):
        self.max_size = max_size
        self.queue = [None] * max_size
        self.front = self.rear = -1

    def is_empty(self):
        return self.front == -1

    def is_full(self):
        return (self.rear + 1) % self.max_size == self.front

    def enqueue(self, item):
        if self.is_full():
            print("Queue is full. Cannot enqueue.")
        elif self.is_empty():
            self.front = self.rear = 0
            self.queue[self.rear] = item
            print(f"Enqueued: {item}")
        else:
            self.rear = (self.rear + 1) % self.max_size
            self.queue[self.rear] = item
            print(f"Enqueued: {item}")

    def dequeue(self):
        if self.is_empty():
            print("Queue is empty. Cannot dequeue.")
            return None
        elif self.front == self.rear:
            item = self.queue[self.front]
            self.front = self.rear = -1
            print(f"Dequeued: {item}")
            return item
        else:
            item = self.queue[self.front]
            self.front = (self.front + 1) % self.max_size
            print(f"Dequeued: {item}")
            return item

    def front_item(self):
        if self.is_empty():
            print("Queue is empty.")
            return None
        return self.queue[self.front]

    def display(self):
        if self.is_empty():
            print("Queue is empty.")
        elif self.rear >= self.front:
            print("Queue:", self.queue[self.front:self.rear+1])
        else:
            print("Queue:", self.queue[self.front:] + self.queue[:self.rear+1])

# Example usage
if __name__ == "__main__":
    cq = CircularQueue(5)
    cq.enqueue(1)
    cq.enqueue(2)
    cq.enqueue(3)
    cq.display()
    cq.dequeue()
    cq.enqueue(4)
    cq.enqueue(5)
    cq.display()
    print("Front:", cq.front_item())