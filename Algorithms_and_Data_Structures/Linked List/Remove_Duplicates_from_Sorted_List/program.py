class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        if not head:
            return None  # If the list is empty, return None

        current = head  # Start with the head of the list

        # Traverse the list
        while current and current.next:
            if current.val == current.next.val:
                # Skip the duplicate node
                current.next = current.next.next
            else:
                # Move to the next distinct element
                current = current.next

        return head  # Return the modified list without duplicates

# Helper function to create a linked list from a list of values
def create_linked_list(values):
    dummy = ListNode(0)
    current = dummy
    for value in values:
        current.next = ListNode(value)
        current = current.next
    return dummy.next

# Helper function to print a linked list
def print_linked_list(head):
    values = []
    while head:
        values.append(head.val)
        head = head.next
    print(" -> ".join(map(str, values)))

# Example usage
list_values = [1, 1, 2]
head = create_linked_list(list_values)

print("Original List:")
print_linked_list(head)

solution = Solution()
modified_head = solution.deleteDuplicates(head)

print("List after removing duplicates:")
print_linked_list(modified_head)
