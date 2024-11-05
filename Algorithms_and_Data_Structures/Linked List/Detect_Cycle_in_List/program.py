class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def hasCycle(self, head: ListNode) -> bool:
        slow, fast = head, head

        # Use Floyd's Cycle-Finding Algorithm
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                return True
        return False

# Helper function to create a linked list with a cycle
def createLinkedList(values, pos):
    head = ListNode(values[0])
    current = head
    nodes = [head]  # store nodes to connect cycle later

    for value in values[1:]:
        new_node = ListNode(value)
        current.next = new_node
        current = new_node
        nodes.append(new_node)

    # Create cycle if pos is not -1
    if pos != -1:
        current.next = nodes[pos]

    return head

# Helper function to display the result
def printResult(head, pos):
    solution = Solution()
    has_cycle = solution.hasCycle(head)
    print(f"Cycle at position {pos}: {'Cycle detected' if has_cycle else 'No cycle detected'}")

# Example usage
values = [3, 2, 0, -4]
pos = 1  # means the last node connects to the node at index 1
head = createLinkedList(values, pos)
print("Input linked list values:", values)
printResult(head, pos)

# Test with no cycle
values_no_cycle = [1, 2]
pos_no_cycle = -1  # no cycle
head_no_cycle = createLinkedList(values_no_cycle, pos_no_cycle)
print("\nInput linked list values:", values_no_cycle)
printResult(head_no_cycle, pos_no_cycle)
