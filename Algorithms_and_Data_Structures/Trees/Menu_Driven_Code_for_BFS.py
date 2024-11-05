from collections import deque

class Node:
    def __init__(self, key):
        # Initialize a new node with the given key, left and right children as None
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
            self.root = Node(key)  # If tree is empty, create the root
        else:
            self._insert_rec(self.root, key)  # Call the recursive insert method

    def _insert_rec(self, root, key):
        # Recursive helper method to insert a key into the tree
        if key < root.val:
            # If key is smaller, go left
            if root.left is None:
                root.left = Node(key)  # Insert new node
            else:
                self._insert_rec(root.left, key)  # Recur on left child
        else:
            # If key is greater or equal, go right
            if root.right is None:
                root.right = Node(key)  # Insert new node
            else:
                self._insert_rec(root.right, key)  # Recur on right child

    def bfs(self):
        if not self.root:
            print("Tree is empty")
            return

        queue = deque([self.root])  # Initialize a queue with the root node
        while queue:
            node = queue.popleft()  # Remove the node from the front of the queue
            print(node.val, end=' ')  # Visit the node

            if node.left:
                queue.append(node.left)  # Add left child to the queue
            if node.right:
                queue.append(node.right)  # Add right child to the queue

def menu():
    tree = BinaryTree()  # Create a new BinaryTree instance
    while True:
        # Display menu options
        print("\n1. Insert\n2. BFS Traversal\n3. Exit")
        choice = int(input("Choose an option: "))
        if choice == 1:
            key = int(input("Enter value to insert: "))  # Get value to insert
            tree.insert(key)  # Insert value into the tree
        elif choice == 2:
            print("BFS Traversal: ", end='')
            tree.bfs()  # Perform and print BFS traversal
        elif choice == 3:
            break  # Exit the loop and program
        else:
            print("Invalid choice!")  # Handle invalid input

if __name__ == "__main__":
    menu()  # Run the menu function if this file is executed
