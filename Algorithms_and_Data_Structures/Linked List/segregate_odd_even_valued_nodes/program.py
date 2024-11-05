class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def oddEvenList(self, head: ListNode) -> ListNode:
        if not head:
            return None
        
        # Initialize odd and even lists
        odd_head = ListNode(0)  # Dummy head for odd list
        even_head = ListNode(0)  # Dummy head for even list
        
        odd = odd_head
        even = even_head
        current = head
        
        # Traverse the original list and segregate based on node values
        while current:
            if current.val % 2 != 0:  # Odd valued node
                odd.next = current
                odd = odd.next
            else:  # Even valued node
                even.next = current
                even = even.next
            current = current.next
        
        # Connect odd list to even list
        odd.next = even_head.next
        even.next = None  # Ensure the last node points to None
        
        return odd_head.next  # Return the head of the modified list

# Helper function to create a linked list from a list
def create_linked_list(arr):
    dummy = ListNode(0)
    current = dummy
    for num in arr:
        current.next = ListNode(num)
        current = current.next
    return dummy.next

# Helper function to print the linked list
def print_linked_list(head):
    values = []
    while head:
        values.append(head.val)
        head = head.next
    print("->".join(map(str, values)))

# Example Usage
input_list = [1, 2, 3, 4, 5]
head = create_linked_list(input_list)
print("Input Linked List:")
print_linked_list(head)

solution = Solution()
segregated_head = solution.oddEvenList(head)

print("Output Linked List:")
print_linked_list(segregated_head)
