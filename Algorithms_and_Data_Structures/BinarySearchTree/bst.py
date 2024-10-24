from typing import List, Optional, Tuple
from bstnode import BSTNode
from tree_exceptions import EmptyTreeError, InvalidValueError

class BST:
    def __init__(self):
        self.root = None
        self.num_nodes = 0

    def insert(self, value: int) -> None:
        """Insert a value into the BST, handling duplicates by incrementing count"""
        if not isinstance(value, (int, float)):
            raise InvalidValueError("Value must be a number")
        
        self.root = self._insert_helper(self.root, value)
        self.num_nodes += 1

    def _insert_helper(self, root: Optional[BSTNode], value: int) -> BSTNode:
        if root is None:
            return BSTNode(value)
        
        if value == root.value:
            root.count += 1
        elif value < root.value:
            root.leftPtr = self._insert_helper(root.leftPtr, value)
        else:
            root.rightPtr = self._insert_helper(root.rightPtr, value)
        
        root.height = 1 + max(self._get_height(root.leftPtr), 
                            self._get_height(root.rightPtr))
        return root

    def delete(self, value: int) -> None:
        """Delete a value from the BST"""
        if self.root is None:
            raise EmptyTreeError("Cannot delete from empty tree")
        self.root = self._delete_helper(self.root, value)
        self.num_nodes -= 1

    def _delete_helper(self, root: Optional[BSTNode], value: int) -> Optional[BSTNode]:
        if root is None:
            return None

        if value < root.value:
            root.leftPtr = self._delete_helper(root.leftPtr, value)
        elif value > root.value:
            root.rightPtr = self._delete_helper(root.rightPtr, value)
        else:
            if root.count > 1:
                root.count -= 1
                return root
            
            if root.leftPtr is None:
                return root.rightPtr
            elif root.rightPtr is None:
                return root.leftPtr
            
            # Node with two children
            successor = self._find_min(root.rightPtr)
            root.value = successor.value
            root.count = successor.count
            successor.count = 1
            root.rightPtr = self._delete_helper(root.rightPtr, successor.value)

        root.height = 1 + max(self._get_height(root.leftPtr), 
                            self._get_height(root.rightPtr))
        return root

    def search(self, value: int) -> Tuple[bool, int]:
        """Search for a value and return (found, count)"""
        node = self._search_helper(self.root, value)
        return (True, node.count) if node else (False, 0)

    def _search_helper(self, root: Optional[BSTNode], value: int) -> Optional[BSTNode]:
        if root is None or root.value == value:
            return root
        
        if value < root.value:
            return self._search_helper(root.leftPtr, value)
        return self._search_helper(root.rightPtr, value)

    def get_height(self) -> int:
        """Get the height of the tree"""
        return self._get_height(self.root)

    def _get_height(self, node: Optional[BSTNode]) -> int:
        return node.height if node else 0

    def _find_min(self, node: BSTNode) -> BSTNode:
        current = node
        while current.leftPtr:
            current = current.leftPtr
        return current

    def get_balance(self, node: Optional[BSTNode]) -> int:
        """Get balance factor of a node"""
        if node is None:
            return 0
        return self._get_height(node.leftPtr) - self._get_height(node.rightPtr)

    def traversals(self) -> dict:
        """Return all three traversals in a dictionary"""
        return {
            'inorder': self.inorder(),
            'preorder': self.preorder(),
            'postorder': self.postorder()
        }

    def inorder(self) -> List[int]:
        """Return inorder traversal as a list"""
        result = []
        self._inorder_helper(self.root, result)
        return result

    def _inorder_helper(self, root: Optional[BSTNode], result: List[int]) -> None:
        if root:
            self._inorder_helper(root.leftPtr, result)
            result.extend([root.value] * root.count)
            self._inorder_helper(root.rightPtr, result)

    def preorder(self) -> List[int]:
        """Return preorder traversal as a list"""
        result = []
        self._preorder_helper(self.root, result)
        return result

    def _preorder_helper(self, root: Optional[BSTNode], result: List[int]) -> None:
        if root:
            result.extend([root.value] * root.count)
            self._preorder_helper(root.leftPtr, result)
            self._preorder_helper(root.rightPtr, result)

    def postorder(self) -> List[int]:
        """Return postorder traversal as a list"""
        result = []
        self._postorder_helper(self.root, result)
        return result

    def _postorder_helper(self, root: Optional[BSTNode], result: List[int]) -> None:
        if root:
            self._postorder_helper(root.leftPtr, result)
            self._postorder_helper(root.rightPtr, result)
            result.extend([root.value] * root.count)
