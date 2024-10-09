class CircularQueue:
    def __init__(self, max_size):
        # Initialize the circular queue with a maximum size
        self.max_size = max_size
        self.queue = [None] * max_size
        self.front = self.rear = -1  # Initialize front and rear pointers

    def is_empty(self):
        # Check if the queue is empty
        return self.front == -1

    def is_full(self):
        # Check if the queue is full
        return (self.rear + 1) % self.max_size == self.front

    def enqueue(self, item):
        if self.is_full():
            print("Queue is full. Cannot enqueue.")
        elif self.is_empty():
            # If queue is empty, set front and rear to 0
            self.front = self.rear = 0
            self.queue[self.rear] = item
            print(f"Enqueued: {item}")
        else:
            # Move rear pointer circularly and add item
            self.rear = (self.rear + 1) % self.max_size
            self.queue[self.rear] = item
            print(f"Enqueued: {item}")

    def dequeue(self):
        if self.is_empty():
            print("Queue is empty. Cannot dequeue.")
            return None
        elif self.front == self.rear:
            # If there's only one element, reset the queue after dequeuing
            item = self.queue[self.front]
            self.front = self.rear = -1
            print(f"Dequeued: {item}")
            return item
        else:
            # Move front pointer circularly and remove item
            item = self.queue[self.front]
            self.front = (self.front + 1) % self.max_size
            print(f"Dequeued: {item}")
            return item

    def front_item(self):
        # Return the front item without removing it
        if self.is_empty():
            print("Queue is empty.")
            return None
        return self.queue[self.front]

    def display(self):
        if self.is_empty():
            print("Queue is empty.")
        elif self.rear >= self.front:
            # If rear hasn't wrapped around, print slice from front to rear
            print("Queue:", self.queue[self.front:self.rear+1])
        else:
            # If rear has wrapped around, print two slices
            print("Queue:", self.queue[self.front:] + self.queue[:self.rear+1])

# Example usage
if __name__ == "__main__":
    cq = CircularQueue(5)  # Create a circular queue with max size 5
    cq.enqueue(1)
    cq.enqueue(2)
    cq.enqueue(3)
    cq.display()
    cq.dequeue()
    cq.enqueue(4)
    cq.enqueue(5)
    cq.display()
    print("Front:", cq.front_item())