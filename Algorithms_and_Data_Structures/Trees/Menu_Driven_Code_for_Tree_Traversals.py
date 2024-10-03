class TreeNode:
    def __init__(self, key):
        # Initialize a new node for the tree
        self.left = None
        self.right = None
        self.val = key

class BinaryTree:
    def __init__(self):
        # Initialize the binary tree with no root
        self.root = None

    def insert(self, key):
        # Public method to insert a new key into the binary tree
        if self.root is None:
            self.root = TreeNode(key)  # Create root if tree is empty
        else:
            self._insert_rec(self.root, key)  # Recur to insert in the correct position

    def _insert_rec(self, root, key):
        # Recursive helper method to insert a key
        if key < root.val:
            if root.left is None:
                root.left = TreeNode(key)  # Insert as left child
            else:
                self._insert_rec(root.left, key)  # Recur on left child
        else:
            if root.right is None:
                root.right = TreeNode(key)  # Insert as right child
            else:
                self._insert_rec(root.right, key)  # Recur on right child

    def inorder(self, node):
        # Inorder traversal: left, root, right
        if node:
            self.inorder(node.left)  # Visit left subtree
            print(node.val, end=' ')  # Visit root
            self.inorder(node.right)  # Visit right subtree

    def preorder(self, node):
        # Preorder traversal: root, left, right
        if node:
            print(node.val, end=' ')  # Visit root
            self.preorder(node.left)  # Visit left subtree
            self.preorder(node.right)  # Visit right subtree

    def postorder(self, node):
        # Postorder traversal: left, right, root
        if node:
            self.postorder(node.left)  # Visit left subtree
            self.postorder(node.right)  # Visit right subtree
            print(node.val, end=' ')  # Visit root

def menu():
    tree = BinaryTree()  # Create a new BinaryTree instance
    while True:
        # Display menu options
        print("\n1. Insert\n2. Inorder Traversal\n3. Preorder Traversal\n4. Postorder Traversal\n5. Exit")
        choice = int(input("Choose an option: "))
        if choice == 1:
            key = int(input("Enter value to insert: "))  # Get value to insert
            tree.insert(key)  # Insert value into the tree
        elif choice == 2:
            print("Inorder Traversal: ", end='')
            tree.inorder(tree.root)  # Perform and print inorder traversal
        elif choice == 3:
            print("Preorder Traversal: ", end='')
            tree.preorder(tree.root)  # Perform and print preorder traversal
        elif choice == 4:
            print("Postorder Traversal: ", end='')
            tree.postorder(tree.root)  # Perform and print postorder traversal
        elif choice == 5:
            break  # Exit the loop and program
        else:
            print("Invalid choice!")  # Handle invalid input

if __name__ == "__main__":
    menu()  # Run the menu function if this file is executed
