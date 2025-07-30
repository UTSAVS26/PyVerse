class ListNode:
    """
    Node for singly-linked list.
    """
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def hasCycle(self, head: ListNode) -> bool:
        """
        Detects a cycle in a linked list using Floyd's Cycle-Finding Algorithm.
        Returns boolean value; True if a cycle exists, False otherwise.
        """
        slow = fast = head

        # Use Floyd's Cycle-Finding Algorithm
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next

            if slow == fast:
                return True

        return False


def createLinkedList(values, pos):
    """
    Helper function to create a linked list that forms a cycle, if pos != -1.
    """

    # To check if the values list is empty
    if len(values) == 0:
        return None

    head = ListNode(values[0])
    current = head
    nodes = [head]  # store nodes to connect cycle later

    for value in values[1:]:
        new_node = ListNode(value)
        current.next = new_node
        current = new_node
        nodes.append(new_node)

    # Create cycle if pos is not -1
    if pos != -1 and 0 <= pos < len(nodes):
        current.next = nodes[pos]

    return head


def printResult(values, pos):
    """
    Creates a linked list and print whether a cycle is detected.
    values: List of integers representing the node values.
    pos: Index in the list where the last node should point to create a cycle. If pos == -1, no cycle is created.
    """
    head = createLinkedList(values, pos)
    solution = Solution()
    has_cycle = solution.hasCycle(head)
    print(f"Cycle at position {pos}: {'Cycle detected' if has_cycle else 'No cycle detected'}")


"""
Example usage
2 parameters are passed in each test case:
- values: A list of integers to create the linked list nodes
- pos: Index at which the last node connects to form a cycle. If -1, no cycle is created
"""
printResult([3, 2, 0, -4], 1)   # In this cycle, last node (-4) points to node at index 1, that is (2)
printResult([1, 2], -1)        # No cycle detected
printResult([], -1)           # Empty list
printResult([1], -1)          # Single node, no cycle detected
printResult([1], 0)           # Single node having a cycle with itself
printResult([1, 2, 3], 0)      # Cycle back to head (1)
printResult([1, 2, 3], 2)      # Cycle back to last node (self-loop)