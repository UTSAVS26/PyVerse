class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        # Using a set to store the nodes of the first linked list
        seen_nodes = set()
        curr = headA
        
        while curr:
            seen_nodes.add(curr)  # Add the current node to the set
            curr = curr.next
        
        curr2 = headB
        while curr2:
            if curr2 in seen_nodes:  # Check if current node is in the set
                return curr2  # Intersection found
            curr2 = curr2.next
        
        return None  # No intersection

# Helper function to create a linked list from a list
def create_linked_list(values):
    if not values:
        return None
    head = ListNode(values[0])
    curr = head
    for value in values[1:]:
        curr.next = ListNode(value)
        curr = curr.next
    return head

# Example usage
if __name__ == "__main__":
    # Create linked lists for the example
    # List A: 1 -> 2 -> 3
    # List B: 4 -> 5
    # Intersection at node with value 3
    intersection_node = ListNode(4)
    
    headA = create_linked_list([1, 2, 3])
    headA.next.next = intersection_node  # Connect intersection
    headB = create_linked_list([5, 6, 7, 8])
    headB.next.next = intersection_node  # Connect intersection

    solution = Solution()
    intersection = solution.getIntersectionNode(headA, headB)

    if intersection:
        print(f"Intersection at node with value: {intersection.val}")
    else:
        print("No intersection")
