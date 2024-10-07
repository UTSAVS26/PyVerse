from bstnode import BSTnode

class BST:
    def __init__(self):
        self.root = None
        self.num_nodes = 0
    
    def insert(self ,value:int) -> None:
        if self.root is None:
            newnode = BSTnode(value)
            self.root = newnode
        else:
           self.insert_helper(self.root ,value)
        self.num_nodes += 1
                    
    def insert_helper(self ,root:BSTnode ,value:int) -> None:
        if value < root.value:
            if root.leftPtr is None:
                root.leftPtr = BSTnode(value)
            else:
                self.insert_helper(root.leftPtr, value)
        else:
            if root.rightPtr is None:
                root.rightPtr = BSTnode(value)
            else:
                self.insert_helper(root.rightPtr, value)
        
    def search(self ,value:int) -> bool:
        temp_node = self.root
        while temp_node is not None:
            if temp_node.value == value:
                return True
            elif temp_node.value > value:
                temp_node = temp_node.leftPtr
            else:
                temp_node = temp_node.rightPtr
        return False

    def inorder(self) -> None:
        self.inorder_helper(self.root)
    
    def inorder_helper(self ,root:BSTnode) -> None:
        if root is not None:
            self.inorder_helper(root.leftPtr)
            print(root.value, end=' ')
            self.inorder_helper(root.rightPtr)
    
    def get_height(self):
        return self.height_helper(self.root)

    def height_helper(self, root: BSTnode) -> int:
        if root is None:
            return -1 
        else:
            left_height = self.height_helper(root.leftPtr)
            right_height = self.height_helper(root.rightPtr)
            return 1 + max(left_height, right_height)
        