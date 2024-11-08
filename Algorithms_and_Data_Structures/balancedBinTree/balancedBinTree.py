# Node class for the binary tree
class Node:
    def __init__(self, val):  # Changed from init to __init__
        self.data = val
        self.left = None
        self.right = None

class Solution:
    # Function to check if a binary tree is balanced
    def isBalanced(self, root):
        return self.dfsHeight(root) != -1

    # Recursive function to calculate the height of the tree
    def dfsHeight(self, root):
        # Base case: if the current node is None, return 0 (height of an empty tree)
        if not root:
            return 0

        # Recursively calculate the height of the left subtree
        left_height = self.dfsHeight(root.left)

        # If the left subtree is unbalanced, propagate the unbalance status
        if left_height == -1:
            return -1

        # Recursively calculate the height of the right subtree
        right_height = self.dfsHeight(root.right)

        # If the right subtree is unbalanced, propagate the unbalance status
        if right_height == -1:
            return -1

        # Check if the difference in height between left and right subtrees is greater than 1
        if abs(left_height - right_height) > 1:
            return -1

        # Return the maximum height of left and right subtrees, adding 1 for the current node
        return max(left_height, right_height) + 1


# Creating a sample binary tree
root = Node(1)
root.left = Node(2)
root.right = Node(3)
root.left.left = Node(4)
root.left.right 
 #Creating an instance of the Solution class
solution = Solution()

# Checking if the tree is balanced
if solution.isBalanced(root):
    print("The tree is balanced.")
else:
    print("The tree is not balanced.")