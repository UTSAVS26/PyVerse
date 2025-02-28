def prec(c):
    if c in ['+', '-']:
        return 1
    if c in ['*', '/']:
        return 2
    return 0

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

def itpr(infix):
    s1 = Stack()  # Stack for operands
    s2 = Stack()  # Stack for operators
    j = 0
    prefix = []

    for i in range(len(infix) - 1, -1, -1):
        t = infix[i]
        if t.isalnum():
            s1.push(t)
        elif t == ')':
            s2.push(')')
        elif t == '(':
            x = s2.pop()
            s1.push(x)
        else:
            if s2.empty():
                s2.push(t)
            else:
                x = s2.pop()
                s1.push(x)
                s2.push(t)

    while not s2.empty():
        x = s2.pop()
        s1.push(x)

    while not s1.empty():
        x = s1.pop()
        if x != ')':
            prefix.append(x)

    prefix.append('\0')  # To indicate the end, if needed
    prefix_str = ''.join(prefix[:-1])  # Exclude the '\0'

    print("\nPrefix expression is...")
    print(prefix_str)

if __name__ == "__main__":
    infix_expr = input("Enter infix expression: ")
    itpr(infix_expr)
