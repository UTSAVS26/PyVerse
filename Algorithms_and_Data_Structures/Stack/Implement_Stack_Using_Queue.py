 # Approach
  '''
  We can solve this using two queues. One queue is to store elements and the other queue is to maintain order. 
  Instead of using list, we can implement this using deque as popleft() is more efficient than pop(). 
  To pop the front element, we need to use pop(0) which takes linear time but deque has popleft() which takes constant time, hence more efficient.
  We can either choose to make push operation or pop operation costly. I choose to make push operation costly.
  '''

from collections import deque
class MyStack:
    def __init__(self):
        self.q1 = deque()
        self.q2 = deque()

    def push(self, x: int) -> None:
        self.q2.append(x)
        while self.q1:
            self.q2.append(self.q1.popleft())
        self.q1, self.q2 = self.q2, self.q1

    def pop(self) -> int:
        return self.q1.popleft()

    def top(self) -> int:
        return self.q1[0]

    def empty(self) -> bool:
        return len(self.q1) == 0

# Change the driver code according to your requirement.
if __name__ == "__main__":
    obj = MyStack()
    obj.push(10)
    obj.push(20)
    print("Top element:", obj.top()) 
    print("Popped element:", obj.pop())
    print("Is stack empty?", obj.empty()) 
    print("Top element:", obj.top())  
    obj.pop()
    print("Is stack empty?", obj.empty())
