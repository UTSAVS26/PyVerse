class QueueUsingStacks:
    def __init__(self):
        self.stack1 = []  # For enqueue operations
        self.stack2 = []  # For dequeue operations

    def is_empty(self):
        # Check if both stacks are empty
        return len(self.stack1) == 0 and len(self.stack2) == 0

    def enqueue(self, item):
        # Add item to stack1 (enqueue operation)
        self.stack1.append(item)
        print(f"Enqueued: {item}")

    def dequeue(self):
        if self.is_empty():
            print("Queue is empty. Cannot dequeue.")
            return None
        
        # If stack2 is empty, transfer all elements from stack1 to stack2
        if not self.stack2:
            while self.stack1:
                self.stack2.append(self.stack1.pop())
        
        # Pop and return the top item from stack2 (which is the front of the queue)
        item = self.stack2.pop()
        print(f"Dequeued: {item}")
        return item

    def front(self):
        if self.is_empty():
            print("Queue is empty.")
            return None
        
        # If stack2 is empty, transfer all elements from stack1 to stack2
        if not self.stack2:
            while self.stack1:
                self.stack2.append(self.stack1.pop())
        
        # Return the top item from stack2 without removing it
        return self.stack2[-1]

    def display(self):
        if self.is_empty():
            print("Queue is empty.")
        else:
            # Display the queue by combining stack2 (reversed) and stack1
            print("Queue:", self.stack2[::-1] + self.stack1)

# Example usage
if __name__ == "__main__":
    # Create a new QueueUsingStacks instance
    q = QueueUsingStacks()
    
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