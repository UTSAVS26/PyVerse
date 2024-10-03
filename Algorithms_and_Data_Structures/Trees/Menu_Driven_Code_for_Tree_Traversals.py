class TreeNode:
    def __init__(self, key):
        self.left = None
        self.right = None
        self.val = key

class BinaryTree:
    def __init__(self):
        self.root = None

    def insert(self, key):
        if self.root is None:
            self.root = TreeNode(key)
        else:
            self._insert_rec(self.root, key)

    def _insert_rec(self, root, key):
        if key < root.val:
            if root.left is None:
                root.left = TreeNode(key)
            else:
                self._insert_rec(root.left, key)
        else:
            if root.right is None:
                root.right = TreeNode(key)
            else:
                self._insert_rec(root.right, key)

    def inorder(self, node):
        if node:
            self.inorder(node.left)
            print(node.val, end=' ')
            self.inorder(node.right)

    def preorder(self, node):
        if node:
            print(node.val, end=' ')
            self.preorder(node.left)
            self.preorder(node.right)

    def postorder(self, node):
        if node:
            self.postorder(node.left)
            self.postorder(node.right)
            print(node.val, end=' ')

def menu():
    tree = BinaryTree()
    while True:
        print("\n1. Insert\n2. Inorder Traversal\n3. Preorder Traversal\n4. Postorder Traversal\n5. Exit")
        choice = int(input("Choose an option: "))
        if choice == 1:
            key = int(input("Enter value to insert: "))
            tree.insert(key)
        elif choice == 2:
            print("Inorder Traversal: ", end='')
            tree.inorder(tree.root)
        elif choice == 3:
            print("Preorder Traversal: ", end='')
            tree.preorder(tree.root)
        elif choice == 4:
            print("Postorder Traversal: ", end='')
            tree.postorder(tree.root)
        elif choice == 5:
            break
        else:
            print("Invalid choice!")

if __name__ == "__main__":
    menu()
