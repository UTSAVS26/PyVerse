class Node:
    """
    Node class for the Dynamic Queue.
    
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


class DynamicQueue:
    """
    Dynamic Queue implementation using linked list structure for efficient memory management.
    """
    def __init__(self):
        """
        Initializes an empty queue with front and rear pointers set to None.
        """
        self.front = None
        self.rear = None

    def enqueue(self, data):
        """
        Adds a new element to the rear of the queue.

        Args:
            data (int): The value to be enqueued.
        """
        n = Node(data)
        if self.front == None:
            # If queue is empty, both front and rear point to the new node.
            self.front = self.rear = n
        else:
            # Add the new node at the rear and update the rear pointer.
            self.rear.next = n
            self.rear = n
        print('\nElement Enqueued in Queue: ', data)

    def dequeue(self):
        """
        Removes an element from the front of the queue.
        """
        if self.front == None:
            # If queue is empty, print a message.
            print('\nQueue is empty..!!')
        else:
            # Remove the front element and move the front pointer to the next node.
            temp = self.front
            self.front = self.front.next
            print('\nElement Dequeued from Queue: ', temp.data)
            # If the queue is now empty, reset the rear to None.
            if self.front == None:
                self.rear = None

    def printQueue(self):
        """
        Prints all elements in the queue.
        """
        if self.front == None:
            # If queue is empty, print a message.
            print('\nQueue is empty..!!')
        else:
            # Traverse from front to rear and print each element.
            temp = self.front
            while temp != None:
                print(temp.data, ' --> ', end='')
                temp = temp.next
            print()


# Main menu-driven code to interact with the dynamic queue.
o = DynamicQueue()

while True:
    print('-----------')
    print('\n1. Enqueue\n2. Dequeue\n3. Print\n0. Exit')
    print('-----------')

    ch = int(input('\nEnter your choice: '))

    if ch == 1:
        # Enqueue operation
        data = int(input('\nEnter value to enqueue in Queue: '))
        o.enqueue(data)

    elif ch == 2:
        # Dequeue operation
        o.dequeue()

    elif ch == 3:
        # Print queue elements
        o.printQueue()

    elif ch == 0:
        # Exit the program
        print('You are out of the program..!!')
        break

    else:
        # Handle incorrect input
        print('\nWrong Input..\nEnter the correct choice..!!\n')
