class Node:
    """
    Node class to represent each element in the linked list.
    
    Args:
        data (int): The value to be stored in the node.
    """
    def __init__(self, data):
        """
        Initializes a new node with the given data and sets the next pointer to None.
        """
        self.data = data
        self.next = None


class LinkedList:
    """
    Linked list implementation with insert, delete, search, and traversal functions.
    """
    def __init__(self):
        """
        Initializes an empty linked list with root set to None.
        """
        self.root = None

    def insertLeft(self, data):
        """
        Inserts an element at the beginning of the linked list.

        Args:
            data (int): The value to be inserted.
        """
        n = Node(data)
        if self.root == None:
            # If the list is empty, make the new node the root.
            self.root = n
        else:
            # Insert the new node at the beginning.
            n.next = self.root
            self.root = n

    def deleteLeft(self):
        """
        Deletes the element from the beginning of the linked list.
        """
        if self.root == None:
            print('\nLinked List is empty..!!')
        else:
            # Remove the first node and update the root to the next node.
            temp = self.root
            self.root = self.root.next
            print('\nDeleted element: ', temp.data)

    def insertRight(self, data):
        """
        Inserts an element at the end of the linked list.

        Args:
            data (int): The value to be inserted.
        """
        n = Node(data)
        if self.root == None:
            # If the list is empty, make the new node the root.
            self.root = n
        else:
            # Traverse to the end of the list and insert the new node.
            temp = self.root
            while temp.next != None:
                temp = temp.next
            temp.next = n

    def deleteRight(self):
        """
        Deletes the element from the end of the linked list.
        """
        if self.root == None:
            print('\nLinked List is empty..!!')
        else:
            # Traverse to the end and remove the last node.
            temp = self.root
            temp2 = self.root
            while temp.next != None:
                temp2 = temp
                temp = temp.next
            temp2.next = None
            if temp == self.root:
                self.root = None
            print('\nDeleted element: ', temp.data)

    def printList(self):
        """
        Prints all elements in the linked list.
        """
        if self.root == None:
            print('\nLinked List is empty..!!')
        else:
            temp = self.root
            print('\nElements in Linked List are: ')
            while temp != None:
                print('|', temp.data, '| -> ', end='')
                temp = temp.next
            print('None\n')

    def searchList(self, data):
        """
        Searches for an element in the linked list.

        Args:
            data (int): The value to search for.
        """
        if self.root == None:
            print('\nLinked List is empty..!!')
        else:
            # Traverse through the list to find the element.
            count = 0
            temp = self.root
            while temp != None:
                if temp.data == data:
                    print('\nElement', data, 'found at node: ', count)
                    return
                temp = temp.next
                count += 1
            print('\nElement not found')

    def deleteElement(self, data):
        """
        Deletes a specific element from the linked list.

        Args:
            data (int): The value to be deleted.
        """
        if self.root == None:
            print('\nLinked List is empty..!!')
        else:
            # Traverse to find the element and remove it from the list.
            count = 0
            temp = self.root
            temp2 = self.root
            while temp != None and temp.data != data:
                temp2 = temp
                temp = temp.next
                count += 1
            if temp != None:
                if temp == self.root:
                    self.root = self.root.next
                elif temp.next == None:
                    temp2.next = None
                else:
                    temp2.next = temp.next
                print('\nDeleted Element:', temp.data, 'from position: ', count)
            else:
                print(data, 'not found in Linked List')


# Menu-driven interaction to perform linked list operations.
o = LinkedList()

while True:
    print('----------------------')
    print('\n1. Insert from Left\n2. Insert from Right\n3. Delete from Left\n4. Delete from Right\n5. Delete Element x\n6. Print Linked List\n7. Search Element x\n0. Exit')
    print('----------------------')

    ch = int(input('\nEnter your choice: '))

    if ch == 1:
        # Insert at the beginning of the list.
        data = int(input('\nEnter value to be inserted in left: '))
        o.insertLeft(data)

    elif ch == 2:
        # Insert at the end of the list.
        data = int(input('\nEnter value to be inserted in right: '))
        o.insertRight(data)

    elif ch == 3:
        # Delete from the beginning of the list.
        o.deleteLeft()

    elif ch == 4:
        # Delete from the end of the list.
        o.deleteRight()

    elif ch == 5:
        # Delete a specific element.
        x = int(input('\nEnter the value of Element x: '))
        o.deleteElement(x)

    elif ch == 6:
        # Print the entire list.
        o.printList()

    elif ch == 7:
        # Search for a specific element.
        data = int(input('Enter the value of Element x: '))
        o.searchList(data)

    elif ch == 0:
        # Exit the program.
        print('You are out of the program..!!')
        break

    else:
        # Handle incorrect input.
        print('\nWrong Input..\nEnter the correct choice..!!\n')
