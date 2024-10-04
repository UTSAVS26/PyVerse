class BSTNode:
    def __init__(self, key):
        # Initialize a new node for Binary Search Tree
        self.left = None
        self.right = None
        self.val = key

class BinarySearchTree:
    def __init__(self):
        # Initialize the Binary Search Tree with no root
        self.root = None

    def insert(self, key):
        # Public method to insert a new key into the BST
        if self.root is None:
            self.root = BSTNode(key)  # Create root if tree is empty
        else:
            self._insert_rec(self.root, key)  # Recur to insert in the correct position

    def _insert_rec(self, root, key):
        # Recursive helper method to insert a key
        if key < root.val:
            if root.left is None:
                root.left = BSTNode(key)  # Insert as left child
            else:
                self._insert_rec(root.left, key)  # Recur on left child
        else:
            if root.right is None:
                root.right = BSTNode(key)  # Insert as right child
            else:
                self._insert_rec(root.right, key)  # Recur on right child

    def search(self, root, key):
        # Public method to search for a key in the BST
        if root is None or root.val == key:
            return root  # Return node if found or None if not found
        if key < root.val:
            return self.search(root.left, key)  # Search left subtree
        return self.search(root.right, key)  # Search right subtree

def menu():
    bst = BinarySearchTree()  # Create a new Binary Search Tree instance
    while True:
        # Display menu options
        print("\n1. Insert\n2. Search\n3. Exit")
        choice = int(input("Choose an option: "))
        if choice == 1:
            key = int(input("Enter value to insert: "))  # Get value to insert
            bst.insert(key)  # Insert value into the BST
        elif choice == 2:
            key = int(input("Enter value to search: "))  # Get value to search
            result = bst.search(bst.root, key)  # Search for the value
            if result:
                print(f"Value {key} found!")  # If found, notify user
            else:
                print(f"Value {key} not found.")  # If not found, notify user
        elif choice == 3:
            break  # Exit the loop and program
        else:
            print("Invalid choice!")  # Handle invalid input

if __name__ == "__main__":
    menu()  # Run the menu function if this file is executed
