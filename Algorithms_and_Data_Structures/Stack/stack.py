class Stack:
    def __init__(self):
        self.items = []

    def is_empty(self):
        """Check if the stack is empty."""
        return len(self.items) == 0

    def push(self, item):
        """Add an item to the top of the stack."""
        self.items.append(item)

    def pop(self):
        """Remove and return the top item from the stack."""
        if not self.is_empty():
            return self.items.pop()
        raise IndexError("Pop from an empty stack")

    def peek(self):
        """Return the top item from the stack without removing it."""
        if not self.is_empty():
            return self.items[-1]
        raise IndexError("Peek from an empty stack")

    def size(self):
        """Return the number of items in the stack."""
        return len(self.items)

# Example usage
if __name__ == "__main__":
    stack = Stack()
    stack.push(10)
    stack.push(20)
    stack.push(30)
    
    print("Top element is:", stack.peek())  # Output: Top element is: 30
    print("Stack size is:", stack.size())    # Output: Stack size is: 3

    print("Popped element:", stack.pop())     # Output: Popped element: 30
    print("Stack size after pop:", stack.size())  # Output: Stack size after pop: 2

    print("Is stack empty?", stack.is_empty())  # Output: Is stack empty? False

    stack.pop()  # Pops 20
    stack.pop()  # Pops 10

    print("Is stack empty after popping all elements?", stack.is_empty())  # Output: Is stack empty after popping all elements? True
