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

def pti(postfix):
    s = Stack()
    
    for t in postfix:
        if t.isalnum():  # Check if the character is an operand
            s.push(t)
        else:  # It's an operator
            operand2 = s.pop()
            operand1 = s.pop()
            infix_expr = f"({operand1} {t} {operand2})"
            s.push(infix_expr)
    
    infix = s.pop()
    
    print("\nPostfix to Infix expression is...")
    print(infix)

if __name__ == "__main__":
    postfix_expr = input("Enter postfix expression: ")
    pti(postfix_expr)
