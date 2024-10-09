from collections import deque

class Deque:
    def __init__(self):
        self.queue = deque()

    def is_empty(self):
        return len(self.queue) == 0

    def add_front(self, item):
        self.queue.appendleft(item)
        print(f"Added to front: {item}")

    def add_rear(self, item):
        self.queue.append(item)
        print(f"Added to rear: {item}")

    def remove_front(self):
        if self.is_empty():
            print("Deque is empty. Cannot remove from front.")
            return None
        item = self.queue.popleft()
        print(f"Removed from front: {item}")
        return item

    def remove_rear(self):
        if self.is_empty():
            print("Deque is empty. Cannot remove from rear.")
            return None
        item = self.queue.pop()
        print(f"Removed from rear: {item}")
        return item

    def front(self):
        if self.is_empty():
            print("Deque is empty.")
            return None
        return self.queue[0]

    def rear(self):
        if self.is_empty():
            print("Deque is empty.")
            return None
        return self.queue[-1]

    def display(self):
        print("Deque:", list(self.queue))

# Example usage
if __name__ == "__main__":
    dq = Deque()
    dq.add_rear(1)
    dq.add_front(2)
    dq.add_rear(3)
    dq.display()
    dq.remove_front()
    dq.remove_rear()
    dq.display()
    print("Front:", dq.front())
    print("Rear:", dq.rear())