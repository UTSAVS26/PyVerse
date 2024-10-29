class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
        self.random = None

# Function to clone the linked list with random pointers
def clone_linked_list_with_random_pointer(head):
    if not head:
        return None

    # Step 1: Create a dictionary to store the mapping from original to cloned nodes
    node_map = {}

    # Step 2: First pass to create all nodes (without setting next or random)
    curr = head
    while curr:
        node_map[curr] = ListNode(curr.val)  # Map original node to its clone
        curr = curr.next

    # Step 3: Second pass to set next and random pointers
    curr = head
    while curr:
        node_map[curr].next = node_map.get(curr.next)      # Set the next pointer
        node_map[curr].random = node_map.get(curr.random)  # Set the random pointer
        curr = curr.next

    # Return the head of the cloned list
    return node_map[head]

# Helper function to create a linked list from a list of values and random indices
def create_linked_list(values, random_indices):
    if not values:
        return None

    # Create all nodes
    nodes = [ListNode(val) for val in values]

    # Set the next pointers
    for i in range(len(nodes) - 1):
        nodes[i].next = nodes[i + 1]

    # Set the random pointers based on random_indices
    for i, random_index in enumerate(random_indices):
        if random_index != -1:  # -1 indicates no random pointer
            nodes[i].random = nodes[random_index]

    return nodes[0]

# Helper function to print the linked list along with random pointers
def print_linked_list(head):
    curr = head
    while curr:
        random_val = curr.random.val if curr.random else "NULL"
        print(f"Node({curr.val}) -> Random({random_val})")
        curr = curr.next

# Main function to demonstrate cloning
if __name__ == "__main__":
    # Define values and random indices for each node
    values = [1, 2, 3, 4, 5]
    random_indices = [2, 4, -1, 0, 1]  # -1 indicates no random pointer

    # Create the linked list with random pointers
    original_list = create_linked_list(values, random_indices)

    print("Original list with random pointers:")
    print_linked_list(original_list)

    # Clone the linked list
    cloned_list = clone_linked_list_with_random_pointer(original_list)

    print("\nCloned list with random pointers:")
    print_linked_list(cloned_list)
