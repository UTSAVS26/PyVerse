class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

# Helper function to check if two trees are mirrors of each other
def is_mirror(left, right):
    if left is None and right is None:
        return True  # Both nodes are None, so they are symmetric
    if left is None or right is None:
        return False  # One of the nodes is None, so they are not symmetric
    # Check if the current nodes have the same value and the left subtree of one side is a mirror of the right subtree of the other side
    return (left.val == right.val) and is_mirror(left.left, right.right) and is_mirror(left.right, right.left)

# Main function to check if a tree is symmetric
def is_symmetric(root):
    if root is None:
        return True  # An empty tree is symmetric
    # Use the helper function to compare the left and right subtrees
    return is_mirror(root.left, root.right)

# Utility function to create a new tree node
def create_node(value):
    return TreeNode(value)

if __name__ == "__main__":
    root = create_node(50)
    root.left = create_node(60)
    root.right = create_node(60)
    root.left.left = create_node(70)
    root.left.right = create_node(80)
    root.right.left = create_node(80)
    root.right.right = create_node(70)
    
    if is_symmetric(root):
        print("The tree is symmetrical")
    else:
        print("The tree is not symmetrical")
