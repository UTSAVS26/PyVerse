class MinStack:
    def __init__(self):
        self.st = []
        self.mini = float('inf')

    def push(self, value: int):
        val = value
        if not self.st:
            self.mini = val
            self.st.append(val)
        else:
            if val < self.mini:
                self.st.append(2 * val - self.mini)
                self.mini = val
            else:
                self.st.append(val)
        print(f"Element Pushed: {value}")

    def pop(self):
        if not self.st:
            print("Stack is empty, cannot pop")
            return
        el = self.st.pop()
        if el < self.mini:
            print(f"Element popped: {self.mini}")
            self.mini = 2 * self.mini - el
        else:
            print(f"Element popped: {el}")

    def top(self) -> int:
        if not self.st:
            print("Stack is empty")
            return -1
        el = self.st[-1]
        if el < self.mini:
            top_element = self.mini
        else:
            top_element = el
        print(f"Top Most Element is: {top_element}")
        return top_element

    def getMin(self) -> int:
        if not self.st:
            print("Stack is empty")
            return -1
        print(f"Minimum Element in the stack is: {self.mini}")
        return self.mini

# Sample input as per the question
stack = MinStack()
stack.push(9)
stack.push(15)
stack.getMin()
stack.push(1)
stack.getMin()
stack.pop()
stack.getMin()
stack.push(4)
stack.getMin()
stack.pop()
stack.top()
