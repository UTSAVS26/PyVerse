class Node:
    """
    Node class for the Doubly Linked List.
    
    Args:
        data (int): The value to be stored in the node.
    """
    def __init__(self, data):
        """
        Initializes a new node with the given data and sets the right and left pointers to None.

        Args:
            data (int): The data to store in the node.
        """
        self.data = data
        self.right = None
        self.left = None


class LinkedList:
    """
    Doubly Linked List implementation with insertion, deletion, and traversal operations.
    """
    def __init__(self):
        """
        Initializes an empty doubly linked list with root and last pointers set to None.
        """
        self.root = None
        self.last = None

    def insertLeft(self, data):
        """
        Inserts a new node at the left (beginning) of the doubly linked list.

        Args:
            data (int): The value to be inserted at the left.
        """
        n = Node(data)
        if self.root == None:
            # If the list is empty, initialize root to the new node.
            self.root = n
        else:
            # Insert the new node at the beginning and update root and the pointers.
            n.right = self.root
            self.root.left = n
            self.root = n

    def deleteLeft(self):
        """
        Deletes a node from the left (beginning) of the doubly linked list.
        """
        if self.root == None:
            print('\nLinked List is empty..!!')
        else:
            temp = self.root
            if self.root.right == self.root.left == None:
                # If only one node is present, set root to None.
                self.root = None
            else:
                # Update root to the next node.
                self.root = self.root.right
                self.root.left = None
            print('Deleted: ', temp.data)

    def insertRight(self, data):
        """
        Inserts a new node at the right (end) of the doubly linked list.

        Args:
            data (int): The value to be inserted at the right.
        """
        n = Node(data)
        if self.root == None:
            # If the list is empty, initialize root to the new node.
            self.root = n
        else:
            # Traverse to the end of the list and insert the new node.
            temp = self.root
            while temp.right != None:
                temp = temp.right
            temp.right = n
            n.left = temp
        print('Inserted Element: ', n.data)

    def deleteRight(self):
        """
        Deletes a node from the right (end) of the doubly linked list.
        """
        if self.root == None:
            print('\nLinked List is empty..!!')
        else:
            if self.root.right == self.root.left:
                # If only one node is present, set root to None.
                self.root = None
            else:
                # Traverse to the last node and delete it.
                temp = self.root
                while temp.right != None:
                    temp = temp.right
                print('\nDeleted: ', temp.data)
                temp = temp.left
                temp.right = None

    def printList(self):
        """
        Prints all elements in the doubly linked list in forward order.
        """
        if self.root == None:
            print('\nLinked List is empty..!!')
            return
        else:
            temp = self.root
            print('Elements of linked list are: ')
            while temp != None:
                print('|', temp.data, ' <-> ', end=" ")
                temp = temp.right
            print('None')
            print('')

    def printListReverse(self):
        """
        Prints all elements in the doubly linked list in reverse order.
        """
        if self.root == None:
            print('\nLinked List is empty..!!')
            return
        else:
            temp = self.root
            # Traverse to the last node
            while temp.right != None:
                temp = temp.right
            # Traverse back to the first node and print elements
            print('Elements of linked list are: ')
            while temp != None:
                print('|', temp.data, ' <-> ', end=" ")
                temp = temp.left
            print('None')
            print('')


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
        o.printListReverse()

    elif ch == 0:
        print('You are out of the program..!!')
        break

    else:
        print('\nWrong Input..\nEnter the correct choice..!!\n')
