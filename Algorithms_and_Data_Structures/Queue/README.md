# Queue Data Structures

A queue is a linear data structure that follows the First-In-First-Out (FIFO) principle. It is an ordered collection of elements where an element is inserted at one end (rear) and removed from the other end (front).

## Types of Queues

### 1. Linear Queue

A linear queue is the basic queue implementation where elements are added at the rear and removed from the front. It has the following operations:
- Enqueue: Add an element to the rear of the queue
- Dequeue: Remove an element from the front of the queue
- Front: Get the front element without removing it
- IsEmpty: Check if the queue is empty
- IsFull: Check if the queue is full (for fixed-size implementations)

### 2. Circular Queue

A circular queue is an improvement over the linear queue that efficiently utilizes memory. It uses a circular array to implement the queue, allowing the rear to wrap around to the beginning of the array when it reaches the end. This solves the problem of unused space in linear queues after dequeuing elements.

### 3. Priority Queue

A priority queue is a special type of queue where each element has a priority associated with it. Elements with higher priority are dequeued before elements with lower priority. If two elements have the same priority, they are served according to their order in the queue.

### 4. Double-Ended Queue (Deque)

A double-ended queue, or deque, is a queue that allows insertion and deletion at both ends. It combines the features of both stacks and queues. Operations include:
- AddFront: Add an element to the front of the deque
- AddRear: Add an element to the rear of the deque
- RemoveFront: Remove an element from the front of the deque
- RemoveRear: Remove an element from the rear of the deque

### 5. Queue Using Two Stacks

This is an implementation of a queue using two stacks. It demonstrates how a queue can be simulated using stack operations. The idea is to use one stack for enqueue operations and another for dequeue operations.

## Applications of Queues

Queues are used in various computer science and real-world applications, including:
1. Task scheduling in operating systems
2. Breadth-First Search (BFS) in graph algorithms
3. Handling of requests on a single shared resource, like a printer or CPU
4. Buffering for data streams
5. Implementing cache in computer systems

Each type of queue has its own use cases and advantages, making them versatile data structures in computer science and software engineering.