class Node:
    """
    Node class for circular doubly linked list.

    Args:
        data (int): The value to be stored in the node.
    """

    def __init__(self, data):
        """
        Initializes a new node with the given data and sets the left and right pointers to None.
        """
        self.data = data
        self.right = None
        self.left = None


class LinkedList:
    """
    Circular Doubly Linked List implementation with various operations.
    """

    def __init__(self):
        """
        Initializes an empty circular doubly linked list with root and last pointers set to None.
        """
        self.root = None
        self.last = None

    def insertLeft(self, data):
        """
        Inserts a new node at the left (beginning) of the circular doubly linked list.

        Args:
            data (int): The value to be inserted at the left.
        """
        n = Node(data)
        if self.root == None:
            # If list is empty, initialize both root and last to the new node.
            self.root = n
            self.last = n
            self.last.right = self.root
            self.root.left = self.last
        else:
            # Insert the new node to the left of root and update pointers.
            n.right = self.root
            self.root.left = n
            self.root = n
            self.last.right = self.root
            self.root.left = self.last
        print('\nInserted Element: ', self.root.data)
        self.printList()

    def deleteLeft(self):
        """
        Deletes a node from the left (beginning) of the circular doubly linked list.
        """
        if self.root == None:
            print('\nLinked List is empty..!!')
        else:
            temp = self.root
            if self.root == self.last:
                # If there is only one node, set root and last to None.
                self.root = None
            else:
                # Delete the root and update pointers.
                self.root = self.root.right
                self.root.left = self.last
                self.last.right = self.root
            print('\nDeleted element: ', temp.data)
        self.printList()

    def insertRight(self, data):
        """
        Inserts a new node at the right (end) of the circular doubly linked list.

        Args:
            data (int): The value to be inserted at the right.
        """
        n = Node(data)
        if self.root == None:
            # If list is empty, initialize both root and last to the new node.
            self.root = n
            self.last = n
            self.last.right = self.root
            self.root.left = self.last
        else:
            # Insert the new node to the right of the last node and update pointers.
            self.last.right = n
            n.left = self.last
            self.last = n
            self.last.right = self.root
            self.root.left = self.last
        print('\nInserted Element: ', n.data)
        self.printList()

    def deleteRight(self):
        """
        Deletes a node from the right (end) of the circular doubly linked list.
        """
        if self.root == None:
            print('\nLinked List is empty..!!')
        else:
            if self.root == self.last:
                # If there is only one node, set root and last to None.
                self.root = None
            else:
                # Delete the last node and update pointers.
                print('Deleted Element: ', self.last.data)
                self.last = self.last.left
                self.last.right = self.root
                self.root.left = self.last
        self.printList()

    def printList(self):
        """
        Prints all elements in the circular doubly linked list in forward order.
        """
        if self.root == None:
            print('\nLinked List is empty..!!')
            return
        temp = self.root
        print('\nElements in Linked List are: ')
        while True:
            print('|', temp.data, '| <-> ', end='')
            temp = temp.right
            if temp == self.root:
                break
        print('Root')
        print()

    def printReverseList(self):
        """
        Prints all elements in the circular doubly linked list in reverse order.
        """
        if self.root == None:
            print('\nLinked List is empty..!!')
            return
        temp = self.last
        print('\nElements in Linked List are: ')
        while True:
            print('|', temp.data, '| <-> ', end='')
            temp = temp.left
            if temp == self.last:
                break
        print('Last')
        print()


# Main menu-driven code to interact with the linked list.
o = LinkedList()

while True:
    print('----------------------')
    print('\n1. Insert from Left\n2. Insert from Right\n3. Delete from Left\n4. Delete from Right\n5. Print Linked List\n6. Print Reverse Linked List\n0. Exit')
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

    elif ch == 6:
        o.printReverseList()

    elif ch == 0:
        print('You are out of the program..!!')
        break

    else:
        print('\nWrong Input..\nEnter the correct choice..!!\n')
