from __future__ import annotations
from typing import TypeVar, Generic, Optional, Any, Iterator

T = TypeVar('T')


class Node(Generic[T]):
    """
    Node class for a doubly linked list.

    Parameters
    ----------
    data : T
        The data to be stored in the node.
    prev : Optional[Node[T]], optional
        Pointer to the previous node, by default None.
    next : Optional[Node[T]], optional
        Pointer to the next node, by default None.
    """
    def __init__(
            self,
            data: T,
            prev: Optional[Node[T]] = None,
            next: Optional[Node[T]] = None
        ) -> None:
        self.data: T = data
        self.prev: Optional[Node[T]] = prev
        self.next: Optional[Node[T]] = next



class LinkedList(Generic[T]):
    """
    Doubly Linked List implementation with insertion, deletion, and traversal operations.
    """
    def __init__(self):
        """
        Initializes an empty doubly linked list with head and last pointers set to None.
        """
        self.head: Optional[Node[T]] = None
        self.tail: Optional[Node[T]] = None
        self.length: int = 0
        self._type: Optional[type] = None 

    def _init_first_node(self, node: Node[T]) -> None:
        """
        Initializes the first node of the linked list.
        
        Parameters
        ----------
        node : Node[T]
            The node to be set as the first node of the linked list.
        """
        self.head = node
        self.tail = node
        self.length = 1
        
    def _check_type(
            self, 
            data: Any
        ) -> None:
        """
        Check whether the given value matches the expected type for the linked list.

        If this is the first insertion, the expected type is set based on the type of the value.
        Subsequent insertions must match this type.

        Parameters
        ----------
        data : Any
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

    def insertLeft(self, data) -> int:
        """
        Inserts a new node at the prev (beginning) of the doubly linked list.

        Parameters
        ----------
        data : T
            The value to be inserted at the prev.
        
        Returns
        ------- 
        int
            The length of the linked list after insertion.
        """
        self._check_type(data)
        node = Node(data)
        if self.head is None:
            self._init_first_node(node)
        else:
            node.next = self.head
            self.head.prev = node
            self.head = node
            self.length += 1
        return len(self)

    def deleteLeft(self) -> Optional[T]:
        """
        Deletes a node from the prev (beginning) of the doubly linked list.

        Returns
        -------
        Optional[T]
            The data of the deleted node, or None if the list is empty.
        
        Raises
        ------
        IndexError
            If the list is empty and a deletion is attempted.
        """
        if self.head is None:
            raise IndexError("Cannot delete from an empty list.")
        temp = self.head
        if self.head is self.tail:
            self.head = None
            self.tail = None
            self.length = 0
        else:
            self.head = self.head.next
            self.head.prev.next = None
            self.head.prev = None
            self.length -= 1
        return temp.data

    def insertRight(self, data: T) -> int:
        """
        Inserts a new node at the next (end) of the doubly linked list.
        
        Parameters
        ----------
        data : T
            The value to be inserted at the next.

        Returns
        -------
        int
            The length of the linked list after insertion.

        Raises
        ------
        TypeError
            If the type of `data` does not match the expected element type.
        """
        self._check_type(data)
        node = Node(data)
        if self.tail is None:
            self._init_first_node(node)
        else:
            node.prev = self.tail
            self.tail.next = node
            self.tail = node
            self.length += 1
        return len(self)

    def deleteRight(self) -> Optional[T]:
        """
        Deletes a node from the next (end) of the doubly linked list.
        
        Returns
        -------
        Optional[T]
            The data of the deleted node, or None if the list is empty.

        Raises
        ------
        IndexError
            If the list is empty and a deletion is attempted.
        """
        if self.tail is None:
            raise IndexError("Cannot delete from an empty list.")
        temp = self.tail
        if self.tail is self.head:
            # If only one node is present, set tail to None.
            self.tail = None
            self.head = None
            self.length = 0
        else:
            # Traverse to the last node and delete it.
            self.tail = self.tail.prev
            self.tail.next.prev = None
            self.tail.next = None
            self.length -= 1
        return temp.data

    def searchlist(self, data: T) -> int:
        """
        Searches for a node with the specified data in the doubly linked list.

        Parameters
        ----------
        data : T
            The value to search for in the linked list.

        Returns
        -------
        int
            The index of the node containing the data if found, otherwise -1.

        Raises
        ------
        TypeError
            If the type of `data` does not match the expected element type.
        """
        self._check_type(data)

        temp = self.head
        pos = 0
        while temp:
            if temp.data == data:
                return pos
            temp = temp.next
            pos += 1
            if temp is self.head:
                break
        return -1
    
    def deleteElement(self, data: T) -> Optional[T]:
        """
        Deletes the first occurrence of an element with the specified data from the doubly linked list.

        Parameters
        ----------
        data : T
            The value to be deleted from the linked list.

        Returns
        -------
        Optional[T]
            The data of the deleted node, or None if the node was not found.

        Raises
        ------
        TypeError
            If the type of `data` does not match the expected element type.
        """
        pos = self.searchlist(data)
        if pos == -1:
            return None
        elif pos == 0:
            return self.deleteLeft()
        elif pos == len(self) - 1:
            return self.deleteRight()
        
        temp = self.head
        for _ in range(pos):
            temp = temp.next

        temp.prev.next = temp.next
        temp.next.prev = temp.prev
        temp.prev = None
        temp.next = None
        self.length -= 1
        return temp.data

    def __str__(self) -> str:
        """
        Returns a string representation of the doubly linked list.

        Returns
        -------
        str
            A string representation of the linked list, showing each node's data.
        """

        elements = []
        temp = self.head
        while temp:
            elements.append(f"[{temp.data}]")
            temp = temp.next
            if temp is self.head:
                break
        return "HEAD -> " + " <-> ".join(elements) + " <- Tail"
    
    def __len__(self) -> int:
        """
        Returns the number of elements in the doubly linked list.

        Returns
        -------
        int
            The length of the linked list.
        """
        return self.length
    
    def __iter__(self): 
        """
        Returns an iterator for the doubly linked list.

        Yields
        ------
        T
            The data of each node in the linked list.
        """
        temp = self.head
        while temp:
            yield temp.data
            temp = temp.next
            if temp is self.head:
                break

    def __contains__(self, item: T) -> bool:
        """
        Checks if an item is present in the doubly linked list.

        Parameters
        ----------
        item : T
            The value to check for presence in the linked list.

        Returns
        -------
        bool
            True if the item is found, False otherwise.
        """
        temp = self.head
        while temp:
            if temp.data == item:
                return True
            temp = temp.next
            if temp is self.head:
                break

        return False

if __name__ == "__main__":
    """
    Main menu-driven code to interact with the doubly linked list.
    """
    # Main menu-driven code to interact with the linked list.
    obj = LinkedList[int]()

    while True:
        print('----------------------')
        print('\n1. Insert from Left\n2. Insert from Right\n3. Delete from Left\n4. Delete from Right\n5. Delete Element x\n6. Print Linked List\n7. Search Element x\n0. Exit')
        print('----------------------')

        try:
            ch = int(input('\nEnter your choice: '))
        except Exception as e:  
            print(f'\nInvalid input.\nError: {e}')
            continue

        try:
            if ch == 1:
                # Check if the input is of the correct type.
                data = int(input('\nEnter value to be inserted in left: '))
                obj.insertLeft(data)

            elif ch == 2: 
                # Check if the input is of the correct type.
                data = int(input('\nEnter value to be inserted in right: '))
                obj.insertRight(data)

            elif ch == 3:
                print(f"Deleted {obj.deleteLeft()} from the beginning of the list.")
            
            elif ch == 4:
                print(f"Deleted {obj.deleteRight()} from the end of the list.")

            elif ch == 5:
                # Delete a specific element.
                x = int(input('\nEnter the value of Element x: '))
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
                data = int(input('Enter the value of Element x: '))
            
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

        except Exception as e :
            print(f"Error: {e}")
            continue
