class Node:
    def __init__(self, key):
        self.left = None
        self.right = None
        self.val = key

class BinaryTree:
    def __init__(self):
        self.root = None

    def insert(self, key):
        if self.root is None:
            self.root = Node(key)
        else:
            self._insert_rec(self.root, key)

    def _insert_rec(self, root, key):
        if key < root.val:
            if root.left is None:
                root.left = Node(key)
            else:
                self._insert_rec(root.left, key)
        else:
            if root.right is None:
                root.right = Node(key)
            else:
                self._insert_rec(root.right, key)

    def inorder(self, node):
        if node:
            self.inorder(node.left)
            print(node.val, end=' ')
            self.inorder(node.right)

def menu():
    tree = BinaryTree()
    while True:
        print("\n1. Insert\n2. Inorder Traversal\n3. Exit")
        choice = int(input("Choose an option: "))
        if choice == 1:
            key = int(input("Enter value to insert: "))
            tree.insert(key)
        elif choice == 2:
            print("Inorder Traversal: ", end='')
            tree.inorder(tree.root)
        elif choice == 3:
            break
        else:
            print("Invalid choice!")

if __name__ == "__main__":
    menu()
