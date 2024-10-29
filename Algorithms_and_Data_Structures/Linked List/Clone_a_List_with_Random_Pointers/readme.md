# Clone a Linked List with Random Pointers

## Problem Statement

Given a linked list where each node contains:
- An integer `val`.
- A `next` pointer to the next node in the list.
- A `random` pointer that points to any node in the list or is `NULL`.

The task is to **create a deep copy (clone) of this linked list**. The cloned list should have the same values, `next`, and `random` pointer connections as the original list but should not share any nodes with it.

## Example

Consider a linked list where:
- Node values: `[1, 2, 3, 4, 5]`
- Random pointers map as follows:
  - `1`'s random points to `3`
  - `2`'s random points to `5`
  - `3` has no random pointer (`NULL`)
  - `4`'s random points to `1`
  - `5`'s random points to `2`

The structure of both the original and cloned list should look like this:

Original list: 1 -> 2 -> 3 -> 4 -> 5 | | | ↓ ↓ ↓ 3 5 1

Cloned list: 1' -> 2' -> 3' -> 4' -> 5' | | | ↓ ↓ ↓ 3' 5' 1'

## Solution Approach

1. **Mapping Original Nodes to Clones**: Use a dictionary to map each node in the original list to its corresponding cloned node, created based on its `val`.
2. **Setting Pointers**: Traverse the list a second time to set the `next` and `random` pointers for each cloned node.
3. **Return Cloned List**: Return the head of the cloned list.

## Steps to Solve

### Step 1: Mapping Original Nodes to Clones
- Use a dictionary (`node_map`) to store a mapping of each node in the original list to its clone, initially creating cloned nodes without setting `next` or `random` pointers.

### Step 2: Setting `next` and `random` Pointers
- Traverse the list again and set the `next` and `random` pointers for each cloned node using the `node_map` dictionary:
  - **Next Pointer**: Map each node's `next` pointer to the corresponding clone.
  - **Random Pointer**: Map each node's `random` pointer to the corresponding clone.

### Step 3: Return the Cloned Head
- Return the cloned head node from `node_map`.

## Code Walkthrough

1. **`clone_linked_list_with_random_pointer(head)`**: Clones the linked list using a dictionary to map each original node to its clone.
2. **`create_linked_list(values, random_indices)`**: Creates a linked list with given `values` and `random` pointers based on the indices in `random_indices`.
3. **`print_linked_list(head)`**: Helper function to print each node’s value and its `random` pointer, if any.


**Explanation of Complexity**
- Time Complexity: 𝑂(𝑛), where n is the number of nodes in the list. We traverse the list twice.
- Space Complexity: 𝑂(𝑛), due to the use of the hash map to store mappings from original to cloned nodes.
