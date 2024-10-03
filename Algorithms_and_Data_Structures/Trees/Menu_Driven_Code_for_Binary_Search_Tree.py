class BSTNode:
    def __init__(self, key):
        self.left = None
        self.right = None
        self.val = key

class BinarySearchTree:
    def __init__(self):
        self.root = None

    def insert(self, key):
        if self.root is None:
            self.root = BSTNode(key)
        else:
            self._insert_rec(self.root, key)

    def _insert_rec(self, root, key):
        if key < root.val:
            if root.left is None:
                root.left = BSTNode(key)
            else:
                self._insert_rec(root.left, key)
        else:
            if root.right is None:
                root.right = BSTNode(key)
            else:
                self._insert_rec(root.right, key)

    def search(self, root, key):
        if root is None or root.val == key:
            return root
        if key < root.val:
            return self.search(root.left, key)
        return self.search(root.right, key)

def menu():
    bst = BinarySearchTree()
    while True:
        print("\n1. Insert\n2. Search\n3. Exit")
        choice = int(input("Choose an option: "))
        if choice == 1:
            key = int(input("Enter value to insert: "))
            bst.insert(key)
        elif choice == 2:
            key = int(input("Enter value to search: "))
            result = bst.search(bst.root, key)
            if result:
                print(f"Value {key} found!")
            else:
                print(f"Value {key} not found.")
        elif choice == 3:
            break
        else:
            print("Invalid choice!")

if __name__ == "__main__":
    menu()
