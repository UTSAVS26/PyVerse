# Intersection of Two Linked Lists

## Problem Statement

In a linked list, the intersection of two lists occurs when two linked lists share a common node. Given two linked lists, this problem aims to find the node where the two lists intersect. If they do not intersect, the result should be `None`.

### Example

Consider the following two linked lists:

- **List A**: 1 -> 2 -> 3
- **List B**: 4 -> 5
- Both lists intersect at the node with value `3`.

The expected output in this case is:


## Approach

To solve this problem, we can use a hash set to store the nodes of the first linked list. As we traverse the second linked list, we can check if any node exists in the set. If we find a match, that node is the intersection point.

### Steps:

1. Traverse the first linked list and add each node to a set.
2. Traverse the second linked list and check if any node is in the set.
3. If a node is found in the set, return that node as the intersection.
4. If no nodes match, return `None`.
