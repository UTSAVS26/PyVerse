class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

def sum_tree(node):
    if node is None:
        return 0
    if node.left is None and node.right is None:
        return node.value

    left_sum = sum_tree(node.left)
    right_sum = sum_tree(node.right)

    if node.value == left_sum + right_sum:
        return node.value + left_sum + right_sum
    else:
        return float('inf')  # Marks as not a sum tree

def is_sum_tree(root):
    return sum_tree(root) != float('inf')

# Example usage
root = Node(60)
root.left = Node(10)
root.right = Node(20)
root.left.left = Node(4)
root.left.right = Node(6)
root.right.left = Node(7)
root.right.right = Node(13)

if is_sum_tree(root):
    print("The tree is a Sum Tree")
else:
    print("The tree is not a Sum Tree")
