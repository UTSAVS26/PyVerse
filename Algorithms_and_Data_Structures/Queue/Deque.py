from collections import deque

class Deque:
    def __init__(self):
        # Initialize the deque using Python's built-in deque class
        self.queue = deque()

    def is_empty(self):
        # Check if the deque is empty
        return len(self.queue) == 0

    def add_front(self, item):
        # Add an item to the front of the deque
        self.queue.appendleft(item)
        print(f"Added to front: {item}")

    def add_rear(self, item):
        # Add an item to the rear of the deque
        self.queue.append(item)
        print(f"Added to rear: {item}")

    def remove_front(self):
        # Remove and return an item from the front of the deque
        if self.is_empty():
            print("Deque is empty. Cannot remove from front.")
            return None
        item = self.queue.popleft()
        print(f"Removed from front: {item}")
        return item

    def remove_rear(self):
        # Remove and return an item from the rear of the deque
        if self.is_empty():
            print("Deque is empty. Cannot remove from rear.")
            return None
        item = self.queue.pop()
        print(f"Removed from rear: {item}")
        return item

    def front(self):
        # Return the front item without removing it
        if self.is_empty():
            print("Deque is empty.")
            return None
        return self.queue[0]

    def rear(self):
        # Return the rear item without removing it
        if self.is_empty():
            print("Deque is empty.")
            return None
        return self.queue[-1]

    def display(self):
        # Display the current state of the deque
        print("Deque:", list(self.queue))

# Example usage
if __name__ == "__main__":
    # Create a new Deque instance
    dq = Deque()
    
    # Add elements to the deque
    dq.add_rear(1)
    dq.add_front(2)
    dq.add_rear(3)
    
    # Display the current state of the deque
    dq.display()
    
    # Remove elements from both ends
    dq.remove_front()
    dq.remove_rear()
    
    # Display the updated state of the deque
    dq.display()
    
    # Check the front and rear elements
    print("Front:", dq.front())
    print("Rear:", dq.rear())