class Stack:
    def __init__(self):
        self.items = []
        self.top = -1

    def is_full(self):
        return self.top == 99  # MAX_SIZE - 1

    def is_empty(self):
        return self.top == -1

    def push(self, item):
        if not self.is_full():
            self.items.append(item)
            self.top += 1

    def pop(self):
        if not self.is_empty():
            self.top -= 1
            return self.items.pop()
        return None

    def peek(self):
        if not self.is_empty():
            return self.items[self.top]
        return None

def is_balanced(expression):
    stack = Stack()
    
    for ch in expression:
        if ch in '({[':
            stack.push(ch)
        elif ch in ')}]':
            if stack.is_empty():
                return False  # Unmatched closing parenthesis
            top = stack.pop()
            if (ch == ')' and top != '(') or \
               (ch == '}' and top != '{') or \
               (ch == ']' and top != '['):
                return False  # Mismatched parentheses

    return stack.is_empty()  # Check if stack is empty at the end

if __name__ == "__main__":
    expression = input("Enter an expression: ")
    if is_balanced(expression):
        print("The parentheses are balanced.")
    else:
        print("The parentheses are not balanced.")
