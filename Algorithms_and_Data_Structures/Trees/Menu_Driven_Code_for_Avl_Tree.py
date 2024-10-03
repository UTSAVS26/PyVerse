class AVLNode:
    def __init__(self, key):
        # Initialize a new node for AVL Tree
        self.key = key
        self.left = None
        self.right = None
        self.height = 1  # Height is initially set to 1

class AVLTree:
    def insert(self, root, key):
        # Public method to insert a new key into the AVL Tree
        if not root:
            return AVLNode(key)  # Create a new node if tree is empty
        elif key < root.key:
            root.left = self.insert(root.left, key)  # Recur to insert in the left subtree
        else:
            root.right = self.insert(root.right, key)  # Recur to insert in the right subtree

        # Update height of this ancestor node
        root.height = 1 + max(self.get_height(root.left), self.get_height(root.right))

        # Check the balance factor
        balance = self.get_balance(root)

        # Left Left Case
        if balance > 1 and key < root.left.key:
            return self.right_rotate(root)  # Right rotate to balance

        # Right Right Case
        if balance < -1 and key > root.right.key:
            return self.left_rotate(root)  # Left rotate to balance

        # Left Right Case
        if balance > 1 and key > root.left.key:
            root.left = self.left_rotate(root.left)  # Left rotate on left child
            return self.right_rotate(root)  # Right rotate to balance

        # Right Left Case
        if balance < -1 and key < root.right.key:
            root.right = self.right_rotate(root.right)  # Right rotate on right child
            return self.left_rotate(root)  # Left rotate to balance

        return root  # Return the (unchanged) node pointer

    def left_rotate(self, z):
        # Perform left rotation
        y = z.right
        T2 = y.left
        y.left = z  # Move z to the left of y
        z.right = T2  # Assign T2 as right child of z
        # Update heights
        z.height = 1 + max(self.get_height(z.left), self.get_height(z.right))
        y.height = 1 + max(self.get_height(y.left), self.get_height(y.right))
        return y  # Return new root

    def right_rotate(self, z):
        # Perform right rotation
        y = z.left
        T3 = y.right
        y.right = z  # Move z to the right of y
        z.left = T3  # Assign T3 as left child of z
        # Update heights
        z.height = 1 + max(self.get_height(z.left), self.get_height(z.right))
        y.height = 1 + max(self.get_height(y.left), self.get_height(y.right))
        return y  # Return new root

    def get_height(self, root):
        # Return height of the node, 0 if None
        if not root:
            return 0
        return root.height

    def get_balance(self, root):
        # Return balance factor of the node
        if not root:
            return 0
        return self.get_height(root.left) - self.get_height(root.right)

    def inorder(self, root):
        # Inorder traversal of the AVL tree
        if root:
            self.inorder(root.left)  # Visit left subtree
            print(root.key, end=' ')  # Visit root
            self.inorder(root.right)  # Visit right subtree

def menu():
    avl = AVLTree()  # Create a new AVL Tree instance
    root = None  # Initialize root as None
    while True:
        # Display menu options
        print("\n1. Insert\n2. Inorder Traversal\n3. Exit")
        choice = int(input("Choose an option: "))
        if choice == 1:
            key = int(input("Enter value to insert: "))  # Get value to insert
            root = avl.insert(root, key)  # Insert value into the AVL tree
        elif choice == 2:
            print("Inorder Traversal: ", end='')
            avl.inorder(root)  # Perform and print inorder traversal
        elif choice == 3:
            break  # Exit the loop and program
        else:
            print("Invalid choice!")  # Handle invalid input

if __name__ == "__main__":
    menu()  # Run the menu function if this file is executed
