from __future__ import annotations
from typing import TypeVar, Generic, Optional, Iterator
from Menu_Driven_Code_for_Linear_LinkedList import LinkedList  

T = TypeVar("T")

class DynamicStack(Generic[T]):
    """
    Dynamic Stack implementation using a linked list structure.
    This class provides methods to push, pop, peek, and print the stack elements.
    """
    def __init__(self):
        """
        Initializes an empty stack using a linked list.
        """
        self._stack = LinkedList[T]()

    def push(self, data: T) -> int:
        """
        Pushes an element onto the stack.

        Parameters
        ----------
        data : T
            The data to be pushed onto the stack.
        
        Returns
        -------
        int
            The new size of the stack after the push operation.
        
        Raises
        ------
        TypeError
            If the data type is not compatible with the stack's type.
        """
        return self._stack.insertLeft(data)


    def pop(self) -> T:
        """
        Pops an element from the stack.

        Returns
        -------
        T
            The data of the popped element.

        Raises
        ------
        IndexError
            If the stack is empty.
        """
        if self._stack.head is None:
            raise IndexError("Cannot pop from an empty stack.")
        return self._stack.deleteLeft()
            

    def peek(self) -> Optional[T]:
        """
        Peeks at the top element of the stack without removing it.

        Returns
        -------
        T
            The data of the top element, or None if the stack is empty.
        """
        if self._stack.head is None:
            raise IndexError("Cannot peek from an empty stack.")
        return self._stack.head.data

    
    def __len__(self) -> int:
        """
        Returns the number of elements in the stack.

        Returns:
            int: The number of elements in the stack.
        """
        return len(self._stack)

    def __iter__(self):
        """
        Returns an iterator for the stack elements.

        Yields
        ------
        T
            The data of each node in the stack.
        """
        return iter(self._stack)

    def __contains__(self, item: T) -> bool:
        """
        Checks if an item is in the stack.

        Parameters
        ----------
        item : T
            The value to check for presence in the stack.

        Returns
        -------
        bool
            True if the item is found, False otherwise.
        """
        return item in self._stack
    
    def __str__(self) -> str:
        """
        Returns a string representation of the stack.

        Returns
        -------
            str
                A string representation of the stack elements.
        """
        return "TOP -> " + " -> ".join(f"[{item}]" for item in self._stack)

if __name__ == "__main__":
    """
    Main function to demonstrate the DynamicStack functionality with a menu-driven interface.
    """
    # Main code with menu-driven interaction
    obj = DynamicStack[int]()

    while True:
        print('-----------')
        print('\n1. Push\n2. Pop\n3. Peek\n4. Print\n0. Exit')
        print('-----------')

        try:
            ch = int(input('\nEnter your choice: '))
        except Exception as e:
            print('ERROR: {e}')
            continue
        if ch == 1:
            # Push operation
            try:
                data = int(input('\nEnter value to push in stack: '))
            except Exception as e:
                print(f"Error: {e}")
                continue

            obj.push(data)

        elif ch == 2:
            # Pop operation
            try:
                ele = obj.pop()
                print(f'\nPopped element: {ele}')
            except Exception as e:
                print(f"Error: {e}")

        elif ch == 3:
            # Peek operation
            try:
                ele = obj.peek()
                print(f'\nTop element: {ele}')
            except Exception as e:
                print(f"Error: {e}")

        elif ch == 4:
            # Print all stack elements
            print("TOP -> ", end="")
            for item in obj:
                print(f'[{item}]', end=' ')
            print("")
        elif ch == 0:
            # Exit the program
            print('You are out of the program..!!')
            break

        else:
            # Handle incorrect input
            print('\nWrong Input..\nEnter the correct choice..!!\n')
