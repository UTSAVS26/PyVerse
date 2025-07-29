from __future__ import annotations
from typing import Generic, TypeVar, Optional, Any
from Menu_Driven_Code_for_Linear_LinkedList import LinkedList, Node

T = TypeVar("T")
class CircularLinkedList(LinkedList[T]):
    """
    A class representing a circular linked list with methods to insert and delete nodes from both ends.
    """
    def __init__(self):
        """
        Initializes an empty circular linked list with root and last pointers set to None.
        """
        super().__init__()
    
    def _init_first_node(self, node: Node[T]) -> None:
        self.head = node
        self.tail = node
        self.tail.next = self.head
        self.length = 1

    def insertLeft(self, data: T) -> int:
        """
        Insert an element at the beginning (left) of the linked list.

        Parameters
        ----------
        data : T
            The value to be inserted at the head of the list.

        Returns
        -------
        int
            The new length of the linked list.

        Raises
        ------
        TypeError
            If the type of `data` does not match the expected element type.
        """
        # Check if the data type matches the expected type.
        self._check_type(data)
        node = Node(data)
        if self.head is None:
            # If the list is empty, initialize the first node.
            self._init_first_node(node)
        else:
            # Insert the new node at the beginning and update head and tail.next.
            node.next = self.head
            self.head = node
            self.tail.next = self.head
            self.length += 1
        return len(self)

    def deleteLeft(self) -> T:
        """
        Deletes the element from the beginning(left) of the linked list.

        Returns
        -------
            T: The removed node's data.

        Raises
        ------
        IndexError
            If trying to pop from an empty linked list.
        """
        if self.head is None:
            raise IndexError("Cannot delete from an empty linked list.")
        
        temp = self.head
        if self.head is self.tail:
            # If there is only one node, set head and tail to None.
            self.tail = self.head = None
            temp.next = None
            self.length = 0
        else:
            # Update head to the next node and adjust the tail.next pointer.
            self.head = self.head.next
            self.tail.next = self.head
            temp.next = None
            self.length -= 1
        return temp.data

    def insertRight(self, data: T) -> int:
        """
        Insert an element at the end (right) of the linked list.

        Parameters
        ----------
        data : T
            The value to be inserted at the end of the list.

        Returns
        -------
        int
            The new length of the linked list.

        Raises
        ------
        TypeError
            If the type of `data` does not match the expected element type.
        
        """
        self._check_type(data)
        node = Node(data)
        if self.head is None:
            self._init_first_node(node)
        else:
            # Insert the new node at the end and update last and last.next.
            self.tail.next = node
            self.tail = node
            self.tail.next = self.head
            self.length += 1
        return len(self)


    def deleteRight(self):
        """
        Deletes the element from the end(right) of the linked list.

        Returns
        -------
        T
            The removed node's data.
        
        Raises
        ------
        IndexError
            If trying to pop from an empty linked list.
        """
        if self.head is None:
            raise IndexError("Cannot delete from an empty linked list.")
    
        temp = self.tail
        if self.head is self.tail:
            # If there is only one node, set head and tail to None.
            self.head = self.tail = None
            temp.next = None
            self.length = 0
        else:
            # Traverse to the node before tail to update tail and its next pointer.
            current = self.head
            while current.next is not self.tail:
                current = current.next
            current.next = self.head
            self.tail = current
            temp.next = None
            self.length -= 1
        return temp.data    

    def searchlist(self, data: T) -> int:
        """
        Searches for a specific element in the linked list.
        
        Parameters
        ----------  
        data : T
            The value to be searched in the linked list.

        Returns
        ------- 
        int
            The position of the element if found, -1 otherwise.

        Raises
        ------
        TypeError
            If the type of `data` does not match the expected element type.
        """
        self._check_type(data)

        if self.length == 0:
            return -1

        current = self.head
        pos = 0 
        while current is not self.tail:
            if current.data == data:
                return pos
            current = current.next
            pos += 1
        if current.data == data:
            return pos
        return -1

    def __str__(self) -> str:
        """
        Returns a string representation of the linked list.

        Returns:
            str: A string representation of the linked list elements.
        """
        if self.head is None:
            return "HEAD ->  <- TAIL"
        elements = []
        current = self.head
        while current is not self.tail :
            elements.append(f"[{current.data}]")
            current = current.next
        elements.append(f"[{self.tail.data}]")
        return "HEAD -> " + " -> ".join(elements) + " <- TAIL"

    def __iter__(self):
        """
        Returns an iterator for the linked list elements.

        Yields
        ------
        T
            The data of each node in the linked list
        """
        current = self.head
        if current is None:
            yield
        while current is not self.tail:
            yield current.data
            current = current.next
        yield current.data

if __name__ == "__main__":

    # Main menu-driven code to interact with the linked list.
    obj = CircularLinkedList[int]()

    while True:
        print('----------------------')
        print('\n1. Insert from Left\n2. Insert from Right\n3. Delete from Left\n4. Delete from Right\n5. Delete Element x\n6. Print Linked List\n7. Search Element x\n0. Exit')
        print('----------------------')

        try:
            ch = int(input('\nEnter your choice: '))
        except Exception as e:  
            print(f'\nInvalid input. Please enter an integer. Error: {e}')
            continue


        if ch == 1:
            try:
                data = int(input('\nEnter value to be inserted in left: '))
            except TypeError as e:
                print(f"Error: {e}")
                continue
            obj.insertLeft(data)

        elif ch == 2:
            try:
                data = int(input('\nEnter value to be inserted in right: '))
            except TypeError as e:
                print(f"Error: {e}")
                continue
            obj.insertRight(data)

        elif ch == 3:
            try:
                print(f"Deleted {obj.deleteLeft()} from the beginning of the list.")
            except IndexError as e:
                print(f"Error: {e}")

        elif ch == 4:
            try:
                print(f"Deleted {obj.deleteRight()} from the end of the list.")
            except IndexError as e:
                print(f"Error: {e}")

        elif ch == 5:
            try:
                x = int(input('\nEnter the value of Element x: '))
            except TypeError as e:
                print(f"Error: {e}")
                continue

            ele = obj.deleteElement(x)
            if ele is None:
                print(f"Element {x} not found in the list.")
            else:
                # If the element was found and deleted.
                print(f"Deleted {ele} from the list.")

        elif ch == 6:
            print(f'\n{str(obj)}')
        
        elif ch == 7:
            # Search for a specific element.        
            try:
                data = int(input('Enter the value of Element x: '))
            except TypeError as e:
                print(f"Error: {e}")
                continue
            index = obj.searchlist(data)
            if index == -1:
                print(f"Element {data} not found.")
            else:
                print(f"Found at Index: {index}")

        elif ch == 0:
            print('You are out of the program..!!')
            break

        else:
            print('\nWrong Input..\nEnter the correct choice..!!\n')
