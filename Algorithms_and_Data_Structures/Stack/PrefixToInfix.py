class Stack:
    def __init__(self):
        self.items = []
    
    def push(self, item):
        self.items.append(item)
    
    def pop(self):
        return self.items.pop() if not self.empty() else None
    
    def top(self):
        return self.items[-1] if not self.empty() else None
    
    def empty(self):
        return len(self.items) == 0

def prti(prefix):
    s = Stack()
    
    # Traverse the prefix expression from right to left
    for t in reversed(prefix):
        if t.isalnum():  # Check if the character is an operand
            s.push(t)
        else:  # It's an operator
            operand1 = s.pop()
            operand2 = s.pop()
            infix_expr = f"({operand1} {t} {operand2})"
            s.push(infix_expr)
    
    infix = s.pop()
    
    print("\nPrefix to Infix expression is...")
    print(infix)

if __name__ == "__main__":
    prefix_expr = input("Enter prefix expression:")
    prti(prefix_expr)
