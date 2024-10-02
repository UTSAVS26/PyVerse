class Node:
    """
    Node class to represent each element in the stack.
    
    Args:
        data (int): The value to be stored in the node.
    """
    def __init__(self, data):
        """
        Initializes a new node with the given data and sets the next pointer to None.

        Args:
            data (int): The data to store in the node.
        """
        self.data = data
        self.next = None


class DynamicStack:
    """
    Dynamic Stack implementation using a linked list structure.
    """
    def __init__(self):
        """
        Initializes an empty stack with tos (Top of Stack) set to None.
        """
        self.tos = None

    def push(self, data):
        """
        Pushes a new element onto the stack.

        Args:
            data (int): The value to be pushed onto the stack.
        """
        n = Node(data)
        if self.tos == None:
            # If stack is empty, set tos to the new node.
            self.tos = n
        else:
            # Otherwise, insert the new node on top and update tos.
            n.next = self.tos
            self.tos = n

    def pop(self):
        """
        Removes the top element from the stack.
        """
        if self.tos == None:
            # If stack is empty, print a message.
            print('\nStack is empty..!!')
        else:
            # Remove the top element and update tos to the next element.
            temp = self.tos
            self.tos = self.tos.next
            print('Popped Element from Stack: ', temp.data)

    def peek(self):
        """
        Returns the top element of the stack without removing it.
        """
        if self.tos == None:
            # If stack is empty, print a message.
            print('\nStack is empty..!!')
        else:
            # Display the top element.
            print('Peeked Element: ', self.tos.data)

    def printStack(self):
        """
        Prints all elements in the stack.
        """
        if self.tos == None:
            # If stack is empty, print a message.
            print('\nStack is empty..!!')
        else:
            # Traverse from top to bottom and print each element.
            print('Stack Data:')
            temp = self.tos
            while temp != None:
                print(temp.data)
                temp = temp.next


# Main code with menu-driven interaction
o = DynamicStack()

while True:
    print('-----------')
    print('\n1. Push\n2. Pop\n3. Peek\n4. Print\n0. Exit')
    print('-----------')

    ch = int(input('\nEnter your choice: '))

    if ch == 1:
        # Push operation
        data = int(input('\nEnter value to push in stack: '))
        o.push(data)

    elif ch == 2:
        # Pop operation
        o.pop()

    elif ch == 3:
        # Peek operation
        o.peek()

    elif ch == 4:
        # Print all stack elements
        o.printStack()

    elif ch == 0:
        # Exit the program
        print('You are out of the program..!!')
        break

    else:
        # Handle incorrect input
        print('\nWrong Input..\nEnter the correct choice..!!\n')
