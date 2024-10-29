# Linked List Cycle Detection

## Problem Statement
Given the `head` of a linked list, determine if the linked list contains a cycle. A cycle exists if a node in the list points back to an earlier node, creating a loop in the list. Internally, a variable `pos` represents the index of the node that the tail's `next` pointer is connected to. If there is no cycle, `pos` is `-1`.

Return `true` if the linked list has a cycle, otherwise return `false`.

### Example

Given the linked list:

- **Input**: `head = [3,2,0,-4], pos = 1`
- **Output**: `true`

### Constraints
- The number of nodes in the list is in the range `[0, 10^4]`.
- `-10^5 <= Node.val <= 10^5`
- `pos` is `-1` if there is no cycle.

## Solution Approach

### Floyd’s Cycle Detection Algorithm (Tortoise and Hare)
This approach uses two pointers (`slow` and `fast`) to detect a cycle:

1. **Initialize**:
   - Set `slow` and `fast` to `head`.
   
2. **Traverse**:
   - Move `slow` by one step (`slow = slow->next`) and `fast` by two steps (`fast = fast->next->next`) in each iteration.
   
3. **Cycle Check**:
   - If there’s a cycle, `slow` and `fast` will eventually meet at some point within the loop.
   - If `fast` reaches the end (`NULL`), there is no cycle.

This algorithm efficiently checks for cycles in `O(n)` time complexity with `O(1)` space complexity.

