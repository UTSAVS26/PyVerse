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

def potpr(postfix):
    s = Stack()
    
    for t in postfix:
        if t.isalnum():  # Check if the character is an operand
            s.push(t)
        else:  # It's an operator
            operand2 = s.pop()
            operand1 = s.pop()
            prefix_expr = f"{t}{operand1}{operand2}"
            s.push(prefix_expr)
    
    prefix = s.pop()
    
    print("\nPostfix to Prefix expression is...")
    print(prefix)

# Example usage:
if __name__ == "__main__":
    postfix_expr = input("Enter postfix expression: ")
    potpr(postfix_expr)
