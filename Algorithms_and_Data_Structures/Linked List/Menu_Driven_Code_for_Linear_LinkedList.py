from __future__ import annotations
from typing import Generic, TypeVar, Optional

T = TypeVar("T")

class Node(Generic[T]):
    """
    Node class to represent each element in the linked list.
    
    Args:
        data (T): The value to be stored in the node.
        next (Optional[Node[T]]): Pointer to the next node in the linked list, defaults to None.
    """
    def __init__(
            self,
            data: T, 
            next: Optional[Node[T]] = None
        ) -> None:
        """
        Initializes a new node with the given data and an optional next node.

        Parameters
        ----------
        data : T
            The value to be stored in the node.
        next : Optional[Node[T]]
            Pointer to the next node in the linked list, defaults to None.
        """
        self.data = data
        self.next = next


class LinkedList(Generic[T]):
    """
    LinkedList class to implement a generic linear singly linked list with common operations.

    This class supports operations to insert elements at both ends, delete elements from both ends,
    search for elements, and print the entire list. It is type-generic and ensures all elements
    added are of the same type, defined by the first insertion.

    Parameters
    ---------------
    T : TypeVar
        The type of data stored in the linked list nodes.

    Attributes
    ----------
    head : Optional[Node[T]]
        The first node in the linked list.
    tail : Optional[Node[T]]
        The last node in the linked list.
    length : int
        The number of elements in the linked list.
    _type : Optional[type]
        The expected type of data stored in the list, inferred from the first insertion.
    """

    def __init__(self) -> None:
        """
        Initializes an empty linked list with head set to None.
        """
        self.head = None
        self.tail = None
        self.length = 0
        self._type = None

    def _init_first_node(self, node: Node[T]) -> None:
        self.head = node
        self.tail = node
        self.length = 1
        
    def _check_type(
            self, 
            data: T
        ) -> None:
        """
        Check whether the given value matches the expected type for the linked list.

        If this is the first insertion, the expected type is set based on the type of the value.
        Subsequent insertions must match this type.

        Parameters
        ----------
            data : T
                The value to be checked for type consistency.

        Raises
        ------
        TypeError
            If the type of `data` does not match the expected element type.
        """

        if self._type is None:
            self._type = type(data) 
        elif not isinstance(data, self._type):
            raise TypeError(f"Expected value of type '{self._type.__name__}', but got '{type(data).__name__}' instead.")

    def insertLeft(
            self, 
            data: T
        ) -> int:
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
            return self.length
        else:
            # Insert the new node at the beginning.
            node.next = self.head
            self.head = node
            self.length += 1
            return self.length

    def deleteLeft(self) -> T:
        """
        Deletes the element from the beginning(left) of the linked list.

        Raises
        ------
        IndexError
            If trying to pop from an empty linked list.

        Returns
        -------
            T: The removed node's data.
        """
        # raise an error if trying to remove from an empty list.
        if self.head is None:
            raise IndexError("Cannot delete from an empty linked list.")
        
        temp = self.head
        if(self.head.next == None):
            # If there's only one node, remove it and set head to None.  
            self.head = None
            self.tail = None
            
        else:
            # Remove the first node and update the head to the next node.
            self.head = self.head.next
            temp.next = None  
        
        self.length -= 1
        return temp.data

    def insertRight(
            self,
            data: T
        ) -> int:
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
            return self.length
        else:
            # Insert the new node at the end.
            self.tail.next = node
            self.tail = node
            self.length += 1
            return self.length
        


    def deleteRight(self) -> T:
        """
        Deletes the element from the end(right) of the linked list.

        Raises
        ------
        IndexError
            If trying to pop from an empty linked list.

        Returns
        -------
        T
            The removed node's data. 
        """
        if self.head is None:
            raise IndexError("Cannot delete from an empty linked list.")
        temp = self.head
        if self.head.next == None:  
            # If there's only one node, remove it and set head to None.
            self.head = None
            self.tail = None
            self.length = 0
            return temp.data
        else:
            # Traverse to the second last node.
            while temp.next.next != None:
                temp = temp.next
            # Remove the last node and update the tail.
            self.tail = temp
            temp = temp.next
            self.tail.next = None
            self.length -= 1
            return temp.data

    def __str__(self) -> str:
        """
        Returns a string representation of the linked list.

        Returns:
            str: A string representation of the linked list elements.
        """
        elements = []
        current = self.head
        while current:
            elements.append(f"[{current.data}]")
            current = current.next

        return "HEAD -> " + " -> ".join(elements) + " <- TAIL"

    def __len__(self) -> int:
        """
        Returns the length of the linked list.

        Returns:
            int: The number of elements in the linked list.
        """
        return self.length
    
    def __contains__(self, item: T) -> bool:
        return self.searchlist(item) != -1
    

    def __iter__(self):
        current = self.head
        while current:
            yield current.data
            current = current.next


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
        if self.length == 0:
            return -1

        self._check_type(data)

        current = self.head
        pos = 0 
        while current:
            if current.data == data:
                return pos
            current = current.next
            pos += 1

        return -1

    def deleteElement(self, data: T) -> Optional[T]:
        """
        Deletes a specific element from the linked list.

        Parameters
        ----------
        data : T
            The value of the element to be deleted.

        Returns
        ------- 
        Optional[T]
            The data of the deleted node, or None if the element was not found.

        Raises
        ------  
        TypeError
            If the type of `data` does not match the expected element type.
        """
        # Check if the data type matches the expected type.
        self._check_type(data)
        # Search for the element in the list.
        pos = self.searchlist(data)
        if pos == -1:
            return None
        if pos == 0:
            # If the element to be deleted is at the head.
            return self.deleteLeft()
        elif pos == self.length - 1:
            # If the element to be deleted is at the tail.
            return self.deleteRight()
        else:
            # If the element to be deleted is in the middle.
            temp = self.head
            for _ in range(pos - 1):
                temp = temp.next
            # Remove the node by skipping it.
            delNode = temp.next
            temp.next = delNode.next
            delNode.next = None
            self.length -= 1
            return delNode.data

if __name__ == "__main__":
    """
    Main function to run the menu-driven code for the linear linked list operations.
    This function allows users to interactively perform operations such as inserting,
    deleting, searching for elements, and printing the entire linked list.
    """

    # Menu-driven interaction to perform linked list operations.
    obj = LinkedList()

    while True:
        print('----------------------')
        print('\n1. Insert from Left\n2. Insert from Right\n3. Delete from Left\n4. Delete from Right\n5. Delete Element x\n6. Print Linked List\n7. Search Element x\n0. Exit')
        print('----------------------')

        try:
            ch = int(input('\nEnter your choice: '))
        except ValueError:
            print("Invalid input. Please enter a number.")
            continue
        except TypeError as e:
            print(f"Error: {e}")
            continue

        if ch == 1:
            # Insert at the beginning of the list.
            data = input('\nEnter value to be inserted in left: ')
            # Check if the input is of the correct type.
            try:
                obj._check_type(data) 
            except TypeError as e:
                print(f"Error: {e}")
                continue
            obj.insertLeft(data)

        elif ch == 2:
            # Insert at the end of the list.
            data = input('\nEnter value to be inserted in right: ')
            try:
                obj._check_type(data)
            except TypeError as e:
                print(f"Error: {e}")
                continue
            obj.insertRight(data)

        elif ch == 3:
            # Delete from the beginning of the list.
            try:
                print(f"Deleted {obj.deleteLeft()} from the beginning of the list.")
            except IndexError as e:
                print(f"Error: {e}")

        elif ch == 4:
            # Delete from the end of the list.
            try:
                print(f"Deleted {obj.deleteRight()} from the end of the list.")
            except Exception as e:
                print(f"Error: {e}")

        elif ch == 5:
            # Delete a specific element.
            x = input('\nEnter the value of Element x: ')
            try:
                obj._check_type(x)
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
            # Print the entire list.
            print(f'\n{str(obj)}')

        elif ch == 7:
            # Search for a specific element.
            data = input('Enter the value of Element x: ')
            try:
                obj._check_type(data) 
            except TypeError as e:
                print(f"Error: {e}")
                continue
            index = obj.searchlist(data)
            if index == -1:
                print(f"Element {data} not found.")
            else:
                print(f"Found at Index: {index}")

        elif ch == 0:
            # Exit the program.
            print('You are out of the program..!!')
            break

        else:
            # Handle incorrect input.
            print('\nWrong Input..\nEnter the correct choice..!!\n')


