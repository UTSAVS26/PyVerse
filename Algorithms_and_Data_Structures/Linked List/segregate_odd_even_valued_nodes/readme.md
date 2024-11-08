# Odd Even Linked List Segregation

## Problem Statement

In this problem, you are given a singly linked list of integers. Your task is to segregate the nodes of the linked list based on their values such that all the nodes with odd values come before the nodes with even values, while maintaining the relative order of the odd and even nodes.

### Example

**Input:** 
1 -> 2 -> 3 -> 4 -> 5
**Output:** 
1 -> 3 -> 5 -> 2 -> 4

## Solution Explanation

The solution involves the following steps:

1. **Initialization**:
   - Create two dummy nodes: one for the odd-valued nodes and one for the even-valued nodes.
   - Use two pointers, `odd` and `even`, to keep track of the last nodes in the respective lists.

2. **Traversal and Segregation**:
   - Traverse the original linked list using a pointer `current`.
   - For each node, check if its value is odd or even:
     - If odd, append it to the odd list.
     - If even, append it to the even list.

3. **Connecting the Lists**:
   - After processing all nodes, connect the end of the odd list to the head of the even list.
   - Ensure the last node of the even list points to `None`.

4. **Return the Result**:
   - The head of the modified linked list is returned by skipping the dummy node used for odd nodes.
