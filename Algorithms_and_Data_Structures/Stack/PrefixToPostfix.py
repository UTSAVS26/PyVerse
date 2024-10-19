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

def prtpo(prefix):
    s = Stack()
    
    for t in reversed(prefix):  # Traverse from right to left
        if t.isalnum():  # Check if the character is an operand
            s.push(t)
        else:  # It's an operator
            operand1 = s.pop()
            operand2 = s.pop()
            postfix_expr = f"{operand1}{operand2}{t}"
            s.push(postfix_expr)
    
    postfix = s.pop()
    
    print("\nPrefix to Postfix expression is...")
    print(postfix)

# Example usage:
if __name__ == "__main__":
    prefix_expr = input("Enter prefix expression: ")
    prtpo(prefix_expr)
