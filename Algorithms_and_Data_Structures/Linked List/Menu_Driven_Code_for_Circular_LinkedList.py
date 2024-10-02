class Node:
    """
    Node class for Circular Linked List.
    
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


class LinkedList:
    """
    Circular Linked List implementation with insertion, deletion, and print operations.
    """
    def __init__(self):
        """
        Initializes an empty circular linked list with root and last pointers set to None.
        """
        self.root = None
        self.last = None

    def insertLeft(self, data):
        """
        Inserts a new node at the left (beginning) of the circular linked list.

        Args:
            data (int): The value to be inserted at the left.
        """
        n = Node(data)
        if self.root == None:
            # If the list is empty, initialize both root and last to the new node.
            self.root = n
            self.last = n
            self.last.next = self.root
        else:
            # Insert the new node at the beginning and update root and last.next.
            n.next = self.root
            self.root = n
            self.last.next = self.root

    def deleteLeft(self):
        """
        Deletes a node from the left (beginning) of the circular linked list.
        """
        if self.root == None:
            print('\nLinked List is empty..!!')
        else:
            temp = self.root
            if self.root == self.last:
                # If there is only one node, set root and last to None.
                self.last = self.root = None
            else:
                # Update root to the next node and adjust the last.next pointer.
                self.root = self.root.next
                self.last.next = self.root
            print('\nDeleted element: ', temp.data)

    def insertRight(self, data):
        """
        Inserts a new node at the right (end) of the circular linked list.

        Args:
            data (int): The value to be inserted at the right.
        """
        n = Node(data)
        if self.root == None:
            # If the list is empty, initialize both root and last to the new node.
            self.root = n
            self.last = n
            self.last.next = self.root
        else:
            # Insert the new node at the end and update last and last.next.
            self.last.next = n
            self.last = n
            self.last.next = self.root

    def deleteRight(self):
        """
        Deletes a node from the right (end) of the circular linked list.
        """
        if self.root == None:
            print('\nLinked List is empty..!!')
        else:
            if self.root == self.last:
                # If there is only one node, set root and last to None.
                self.root = self.last = None
            else:
                # Traverse to the second-last node and update last to the second-last node.
                temp = self.root
                temp2 = self.root
                while temp.next != self.root:
                    temp2 = temp
                    temp = temp.next
                self.last = temp2
                temp2.next = self.root
            print('\nDeleted element: ', temp.data)

    def printList(self):
        """
        Prints all elements in the circular linked list in forward order.
        """
        if self.root == None:
            print('\nLinked List is empty..!!')
            return
        temp = self.root
        print('\nElements in Linked List are: ')
        while True:
            print('|', temp.data, '| -> ', end='')
            temp = temp.next
            if temp == self.root:
                break
        print('None')
        print()


# Main menu-driven code to interact with the linked list.
o = LinkedList()

while True:
    print('----------------------')
    print('\n1. Insert from Left\n2. Insert from Right\n3. Delete from Left\n4. Delete from Right\n5. Print Linked List\n0. Exit')
    print('----------------------')

    ch = int(input('\nEnter your choice: '))

    if ch == 1:
        data = int(input('\nEnter value to be inserted in left: '))
        o.insertLeft(data)

    elif ch == 2:
        data = int(input('\nEnter value to be inserted in right: '))
        o.insertRight(data)

    elif ch == 3:
        o.deleteLeft()

    elif ch == 4:
        o.deleteRight()

    elif ch == 5:
        o.printList()

    elif ch == 0:
        print('You are out of the program..!!')
        break

    else:
        print('\nWrong Input..\nEnter the correct choice..!!\n')
