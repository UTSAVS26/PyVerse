from __future__ import annotations
from typing import TypeVar, Generic, Optional
from Menu_Driven_Code_for_Linear_LinkedList import LinkedList  # assuming it's imported from your module

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
        try:
            return self._stack.deleteLeft()
        except IndexError:
            raise IndexError("Cannot pop from an empty stack.")

    def peek(self) -> Optional[T]:
        """
        Peeks at the top element of the stack without removing it.

        Returns
        -------
        Optional[T]
            The data of the top element, or None if the stack is empty.
        """
        if len(self._stack) == 0:
            raise IndexError("Cannot peek from an empty stack.")
        return self._stack.head.data

    
    def __len__(self) -> int:
        """
        Returns the number of elements in the stack.
        """
        return len(self._stack)

    def __iter__(self):
        """
        Returns an iterator for the stack elements.
        """
        return iter(self._stack)

    def __contains__(self, item: T) -> bool:
        """
        Checks if an item is in the stack.
        """
        return item in self._stack
    
    def __str__(self) -> str:
        """
        Returns a string representation of the stack.
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
        except TypeError:
            print('\nInvalid input. Please enter an integer.')
            continue
        if ch == 1:
            # Push operation
            data = int(input('\nEnter value to push in stack: '))
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
