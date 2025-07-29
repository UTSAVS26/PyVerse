from __future__ import annotations
from typing import TypeVar, Generic, Optional, Iterator
from Menu_Driven_Code_for_Linear_LinkedList import LinkedList

T = TypeVar('T')

class DynamicQueue(Generic[T]):
    """
    Dynamic Queue implementation using a linked list.
    This class supports enqueue, dequeue, peek, and printing operations.
    """

    def __init__(self) -> None:
        """
        Initializes an empty queue.
        """
        self._queue = LinkedList[T]()

    def enqueue(self, data: T) -> int:
        """
        Adds an element to the rear of the queue.

        Parameters
        ----------
        data : T
            The data to enqueue.

        Returns
        -------
        int
            New length of the queue after insertion.

        Raises
        ------
        TypeError
            If the data type does not match the queue's type.
        """
        return self._queue.insertRight(data)

    def dequeue(self) -> T:
        """
        Removes and returns the front element from the queue.

        Returns
        -------
        T
            The dequeued element.

        Raises
        ------
        IndexError
            If the queue is empty.
        """
        if self._queue.head is None:
            raise IndexError("Cannot dequeue from an empty queue.")
        return self._queue.deleteLeft()
    def peek(self) -> Optional[T]:
        """
        Returns the front element of the queue without removing it.

        Returns
        -------
        Optional[T]
            The front element, or None if the queue is empty.

        Raises
        ------
        IndexError
            If the queue is empty.
        """
        if self._queue.head is None:
            raise IndexError("Cannot peek from an empty queue.")
        return self._queue.head.data

    def __len__(self) -> int:
        """
        Returns the number of elements in the queue.

        Returns
        ------- 
        int
            The number of elements currently in the queue.
        """
        return len(self._queue)

    def __iter__(self):
        """
        Returns an iterator over the elements of the queue.

        Returns
        -------
        Iterator[T]
            An iterator that allows iteration over the queue elements.
        """
        return iter(self._queue)

    def __contains__(self, item: T) -> bool:
        """
        Checks if an item exists in the queue.

        Parameters
        ----------
        item : T
            The item to check for existence in the queue.

        Returns
        -------
        bool
            True if the item is in the queue, False otherwise.

        Raises
        ------
        TypeError
            If the item type does not match the queue's type.
        """
        return item in self._queue

    def __str__(self) -> str:
        """
        Returns a string representation of the queue.

        Returns
        -------
        str
            A string showing the front and rear of the queue with elements in between.
        """
        return "FRONT -> " + " -> ".join(f"[{item}]" for item in self._queue) + " <- REAR"

if __name__ == "__main__":

    # Main menu-driven code to interact with the dynamic queue.
    obj = DynamicQueue[int]()

    while True:
        print('-----------')
        print('\n1. Enqueue\n2. Dequeue\n3. Print\n0. Exit')
        print('-----------')

        try:
            ch = int(input('\nEnter your choice: '))
        except Exception as e:
            print(f'Error: {e}')
            continue


        if ch == 1:
            # Enqueue operation
            try:
                data = int(input('\nEnter value to enqueue in Queue: '))
                obj.enqueue(data)
                print(f"\nElement Enqueued in Queue: {data}")
            except Exception as e:
                print(f"Error: {e}")

        elif ch == 2:
            # Dequeue operation
            try:
                data = obj.dequeue()
                print(f"\nElement Dequeued from Queue: {data}")
            except Exception as e:
                print(f"Error: {e}")

        elif ch == 3:
            # Print queue elements
            print("FRONT -> ", end="")
            for item in obj:
                print(f"[{item}]", end=" ")
            print("<- REAR")

        elif ch == 4:
            # Peek operation
            try:
                data = obj.peek()
                print(f"\nFront of Queue: {data}")
            except Exception as e:
                print(f"Error: {e}")
        
        elif ch == 0:
            # Exit the program
            print('You are out of the program..!!')
            break

        else:
            # Handle incorrect input
            print('\nWrong Input..\nEnter the correct choice..!!\n')
