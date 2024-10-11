class LinearQueue:
    def __init__(self, max_size):
        # Initialize an empty list to store queue elements
        self.queue = []
        # Set the maximum size of the queue
        self.max_size = max_size

    def is_empty(self):
        # Check if the queue is empty
        return len(self.queue) == 0

    def is_full(self):
        # Check if the queue has reached its maximum size
        return len(self.queue) == self.max_size

    def enqueue(self, item):
        # Add an item to the rear of the queue
        if self.is_full():
            print("Queue is full. Cannot enqueue.")
        else:
            self.queue.append(item)
            print(f"Enqueued: {item}")

    def dequeue(self):
        # Remove and return an item from the front of the queue
        if self.is_empty():
            print("Queue is empty. Cannot dequeue.")
            return None
        else:
            item = self.queue.pop(0)  # Remove the first item (index 0)
            print(f"Dequeued: {item}")
            return item

    def front(self):
        # Return the front item without removing it
        if self.is_empty():
            print("Queue is empty.")
            return None
        return self.queue[0]  # Return the first item without removing it

    def display(self):
        # Display the current state of the queue
        print("Queue:", self.queue)

# Example usage
if __name__ == "__main__":
    # Create a new LinearQueue instance with a maximum size of 5
    q = LinearQueue(5)
    
    # Add elements to the queue
    q.enqueue(1)
    q.enqueue(2)
    q.enqueue(3)
    
    # Display the current state of the queue
    q.display()
    
    # Remove an element from the front of the queue
    q.dequeue()
    
    # Display the updated state of the queue
    q.display()
    
    # Check the front element
    print("Front:", q.front())