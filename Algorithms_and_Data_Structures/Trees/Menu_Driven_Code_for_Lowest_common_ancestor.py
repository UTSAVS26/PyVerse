class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None


class BinaryTree:
    def __init__(self):
        self.root = None

    def add_node(self, value, parent_value, is_left):
        # Add a node to the tree
        if self.root is None:
            self.root = TreeNode(value)
        else:
            parent_node = self._find_node(self.root, parent_value)
            if parent_node:
                new_node = TreeNode(value)
                if is_left:
                    parent_node.left = new_node
                else:
                    parent_node.right = new_node
            else:
                print("Parent node not found.")

    def _find_node(self, current, value):
        # Helper function to find a node with a given value
        if current is None:
            return None
        if current.value == value:
            return current
        left = self._find_node(current.left, value)
        if left:
            return left
        return self._find_node(current.right, value)

    def lca(self, root, n1, n2):
        # Function to find LCA of n1 and n2
        if root is None:
            return None
        if root.value == n1 or root.value == n2:
            return root

        left_lca = self.lca(root.left, n1, n2)
        right_lca = self.lca(root.right, n1, n2)

        if left_lca and right_lca:
            return root

        return left_lca if left_lca else right_lca

    def find_lca(self, n1, n2):
        # Public method to find LCA of two nodes
        lca_node = self.lca(self.root, n1, n2)
        if lca_node:
            return lca_node.value
        else:
            return None


def menu():
    bt = BinaryTree()  # Create a new BinaryTree instance
    while True:
        print("\n1. Add Node\n2. Find LCA\n3. Exit")
        choice = int(input("Choose an option: "))

        if choice == 1:
            value = int(input("Enter node value: "))
            if bt.root is None:
                # If the tree is empty, make this the root node
                bt.add_node(value, None, None)
                print(f"Node {value} added as root.")
            else:
                parent_value = int(input("Enter parent node value: "))
                is_left = input("Is this a left child? (y/n): ").lower() == 'y'
                bt.add_node(value, parent_value, is_left)
                print(f"Node {value} added as {'left' if is_left else 'right'} child of {parent_value}.")

        elif choice == 2:
            n1 = int(input("Enter first node value: "))
            n2 = int(input("Enter second node value: "))
            lca_value = bt.find_lca(n1, n2)
            if lca_value is not None:
                print(f"The Lowest Common Ancestor of {n1} and {n2} is: {lca_value}")
            else:
                print(f"One or both nodes not found in the tree.")

        elif choice == 3:
            print("Exiting...")
            break

        else:
            print("Invalid choice!")


if __name__ == "__main__":
    menu()
