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

def itp(infix):
    m = Stack()
    j = 0
    postfix = []
    
    for t in infix:
        if t.isalnum():
            postfix.append(t)
        elif t == '(':
            m.push('(')
        elif t == ')':
            while (x := m.pop()) != '(':
                postfix.append(x)
        else:
            while not m.empty() and prec(t) <= prec(m.top()):
                postfix.append(m.pop())
            m.push(t)
    
    while not m.empty():
        postfix.append(m.pop())
    
    postfix.append('\0')  # To indicate the end, if needed
    postfix_str = ''.join(postfix[:-1])  # Exclude the '\0'
    
    print("\nInfix to Postfix:")
    print(postfix_str)

if __name__ == "__main__":
    infix=input("Enter infix Expression: ")
    itp(infix)
