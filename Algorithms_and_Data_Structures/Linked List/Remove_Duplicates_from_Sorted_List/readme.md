# Remove Duplicates from Sorted List

## Problem Statement

Given the head of a sorted linked list, remove all duplicates so that each element appears only once. Return the modified linked list, which remains sorted.

### Example

**Input:**
- `head = [1, 1, 2]`

**Output:**
- `[1, 2]`

### Constraints
- The linked list is sorted in non-decreasing order.
- The output should retain the sorted order with no duplicate values.

## Solution Explanation

The solution involves modifying the linked list in place to remove duplicate values:
1. **Initialize a `current` Pointer**:
   - Start with a pointer `current` at the head of the list.
2. **Single Pass through the List**:
   - Traverse the list, checking each node's value against the value of the next node.
   - If `current.val` equals `current.next.val`, we skip the duplicate by adjusting the `next` pointer of `current` to `current.next.next`.
   - If the values differ, simply move `current` to the next node.
3. **In-place Modification**:
   - This approach operates directly on the linked list, so no extra data structures are needed.
   
### Complexity
- **Time Complexity**: `O(n)`, where `n` is the number of nodes in the linked list, since we traverse each node once.
- **Space Complexity**: `O(1)`, as the list is modified in place without additional storage.