from typing import Optional

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def searchBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        if not root:
            return None
        while root:
            if root.val == val:
                return root
            elif val < root.val:
                root = root.left
            else:
                root = root.right
        return None

def print_subtree(node):
    if node:
        print(node.val, end=' ')
        print_subtree(node.left)
        print_subtree(node.right)

# Driver Code - Change it according to your requirement
root = TreeNode(4)
root.left = TreeNode(2, TreeNode(1), TreeNode(3))
root.right = TreeNode(7)

sol = Solution()
result = sol.searchBST(root, 2)

print_subtree(result)
