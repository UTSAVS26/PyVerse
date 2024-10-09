import heapq

class PriorityQueue:
    def __init__(self):
        # Initialize an empty list to store queue elements
        self.queue = []
        # Index to maintain FIFO order for items with the same priority
        self.index = 0

    def is_empty(self):
        # Check if the queue is empty
        return len(self.queue) == 0

    def enqueue(self, item, priority):
        # Add an item to the queue with a given priority
        # We use negative priority to turn min-heap into max-heap
        heapq.heappush(self.queue, (-priority, self.index, item))
        self.index += 1
        print(f"Enqueued: {item} with priority {priority}")

    def dequeue(self):
        # Remove and return the item with the highest priority
        if self.is_empty():
            print("Queue is empty. Cannot dequeue.")
            return None
        else:
            # Ignore priority and index, return only the item
            _, _, item = heapq.heappop(self.queue)
            print(f"Dequeued: {item}")
            return item

    def front(self):
        # Return the item with the highest priority without removing it
        if self.is_empty():
            print("Queue is empty.")
            return None
        return self.queue[0][2]  # Return only the item, not priority or index

    def display(self):
        # Display the current state of the queue
        # Convert back to positive priority for display
        print("Queue:", [(item, -priority) for priority, _, item in self.queue])

# Example usage
if __name__ == "__main__":
    # Create a new PriorityQueue instance
    pq = PriorityQueue()
    
    # Add elements to the queue with different priorities
    pq.enqueue("Task 1", 2)
    pq.enqueue("Task 2", 1)
    pq.enqueue("Task 3", 3)
    
    # Display the current state of the queue
    pq.display()
    
    # Remove the highest priority element
    pq.dequeue()
    
    # Display the updated state of the queue
    pq.display()
    
    # Check the front element (highest priority)
    print("Front:", pq.front())