class QueueUsingStacks:
    def __init__(self):
        self.stack1 = []  # For enqueue
        self.stack2 = []  # For dequeue

    def is_empty(self):
        return len(self.stack1) == 0 and len(self.stack2) == 0

    def enqueue(self, item):
        self.stack1.append(item)
        print(f"Enqueued: {item}")

    def dequeue(self):
        if self.is_empty():
            print("Queue is empty. Cannot dequeue.")
            return None
        
        if not self.stack2:
            while self.stack1:
                self.stack2.append(self.stack1.pop())
        
        item = self.stack2.pop()
        print(f"Dequeued: {item}")
        return item

    def front(self):
        if self.is_empty():
            print("Queue is empty.")
            return None
        
        if not self.stack2:
            while self.stack1:
                self.stack2.append(self.stack1.pop())
        
        return self.stack2[-1]

    def display(self):
        if self.is_empty():
            print("Queue is empty.")
        else:
            print("Queue:", self.stack2[::-1] + self.stack1)

# Example usage
if __name__ == "__main__":
    q = QueueUsingStacks()
    q.enqueue(1)
    q.enqueue(2)
    q.enqueue(3)
    q.display()
    q.dequeue()
    q.display()
    print("Front:", q.front())