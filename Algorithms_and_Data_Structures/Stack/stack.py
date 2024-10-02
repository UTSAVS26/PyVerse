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
    # Example with integers
    int_stack = Stack()
    int_stack.push(10)
    int_stack.push(20)
    int_stack.push(30)
    print("Top element is:", int_stack.peek())  # Output: Top element is: 30
    print("Stack size is:", int_stack.size())  # Output: Stack size is: 3
    print("Popped element:", int_stack.pop())  # Output: Popped element: 30
    print("Stack size after pop:", int_stack.size())  # Output: Stack size after pop: 2
    print("Is stack empty?", int_stack.is_empty())  # Output: Is stack empty? False

    # Pop remaining elements
    int_stack.pop()  # Pops 20
    int_stack.pop()  # Pops 10
    print("Is stack empty after popping all elements?", int_stack.is_empty())  # Output: Is stack empty after popping all elements? True

    # Example with strings
    string_stack = Stack()
    string_stack.push("Hello")
    string_stack.push("World")
    print("Top element is:", string_stack.peek())  # Output: Top element is: World
    print("Stack size is:", string_stack.size())  # Output: Stack size is: 2
    print("Popped element:", string_stack.pop())  # Output: Popped element: World
    print("Stack size after pop:", string_stack.size())  # Output: Stack size after pop: 1
    print("Is stack empty?", string_stack.is_empty())  # Output: Is stack empty? False



# Top element is: 30
# Stack size is: 3
# Popped element: 30
# Stack size after pop: 2
# Is stack empty? False
# Is stack empty after popping all elements? True
# Top element is: World
# Stack size is: 2
# Popped element: World
# Stack size after pop: 1
# Is stack empty? False