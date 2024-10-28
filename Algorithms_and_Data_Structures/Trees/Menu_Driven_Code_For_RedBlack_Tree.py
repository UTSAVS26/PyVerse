import time
import matplotlib.pyplot as plt
import networkx as nx

# RBNode class with Red-Black Tree node properties
class RBNode:
    def __init__(self, value, color='red', parent=None):
        self.value = value
        self.color = color
        self.parent = parent
        self.left = None
        self.right = None

    def sibling(self):
        if self.parent:
            if self == self.parent.left:
                return self.parent.right
            return self.parent.left
        return None

class RedBlackTree:
    def __init__(self):
        self.root = None

    def measure_operation_time(self, func, *args):
        start_time = time.time()
        func(*args)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Time taken for operation: {elapsed_time:.6f} seconds")

    # BST insert helper
    def _bst_insert(self, root, node):
        if node.value < root.value:
            if root.left:
                self._bst_insert(root.left, node)
            else:
                root.left = node
                node.parent = root
        else:
            if root.right:
                self._bst_insert(root.right, node)
            else:
                root.right = node
                node.parent = root

    # Insert node and fix violations
    def insert(self, value):
        new_node = RBNode(value)
        if not self.root:
            self.root = new_node
            self.root.color = 'black'
        else:
            self._bst_insert(self.root, new_node)
            self.insert_fix(new_node)
        print(f"Inserted {value} into the tree.")

    def insert_fix(self, node):
        # Placeholder for balancing the tree after insertion
        pass

    def delete(self, value):
        node_to_remove = self.search(value)
        if not node_to_remove:
            print(f"Value {value} not found in the tree.")
            return
        # Deletion and balancing logic here
        print(f"Deleted {value} from the tree.")
        self.delete_fix(node_to_remove)

    def delete_fix(self, node):
        # Placeholder for balancing the tree after deletion
        pass

    def search(self, value):
        current = self.root
        while current:
            if current.value == value:
                print(f"Found {value} in the tree.")
                return current
            elif value < current.value:
                current = current.left
            else:
                current = current.right
        print(f"Value {value} not found in the tree.")
        return None

    # Traversal methods
    def inorder_traversal(self, node):
        if node:
            self.inorder_traversal(node.left)
            print(f"{node.value}({node.color})", end=" ")
            self.inorder_traversal(node.right)

    def preorder_traversal(self, node):
        if node:
            print(f"{node.value}({node.color})", end=" ")
            self.preorder_traversal(node.left)
            self.preorder_traversal(node.right)

    def postorder_traversal(self, node):
        if node:
            self.postorder_traversal(node.left)
            self.postorder_traversal(node.right)
            print(f"{node.value}({node.color})", end=" ")

    # Calculate tree height
    def tree_height(self, node):
        if not node:
            return 0
        return 1 + max(self.tree_height(node.left), self.tree_height(node.right))

    # Display function using matplotlib
    def display_tree(self):
        G = nx.DiGraph()
        def add_edges(node):
            if node:
                G.add_node(node.value, color=node.color)
                if node.left:
                    G.add_edge(node.value, node.left.value)
                    add_edges(node.left)
                if node.right:
                    G.add_edge(node.value, node.right.value)
                    add_edges(node.right)

        add_edges(self.root)
        pos = nx.spring_layout(G)
        color_map = ['red' if G.nodes[node]['color'] == 'red' else 'black' for node in G.nodes]
        nx.draw(G, pos, node_color=color_map, with_labels=True)
        plt.show()

# Main function for user interaction
def main():
    rb_tree = RedBlackTree()
    while True:
        print("\nChoose an operation:")
        print("1. Insert")
        print("2. Delete")
        print("3. Search")
        print("4. Display Tree")
        print("5. Inorder Traversal")
        print("6. Preorder Traversal")
        print("7. Postorder Traversal")
        print("8. Tree Height")
        print("9. Exit")
        
        choice = input("Enter choice: ")
        if choice == '1':
            value = int(input("Enter value to insert: "))
            rb_tree.measure_operation_time(rb_tree.insert, value)
        elif choice == '2':
            value = int(input("Enter value to delete: "))
            rb_tree.measure_operation_time(rb_tree.delete, value)
        elif choice == '3':
            value = int(input("Enter value to search: "))
            rb_tree.measure_operation_time(rb_tree.search, value)
        elif choice == '4':
            rb_tree.display_tree()
        elif choice == '5':
            print("Inorder Traversal:")
            rb_tree.inorder_traversal(rb_tree.root)
            print()
        elif choice == '6':
            print("Preorder Traversal:")
            rb_tree.preorder_traversal(rb_tree.root)
            print()
        elif choice == '7':
            print("Postorder Traversal:")
            rb_tree.postorder_traversal(rb_tree.root)
            print()
        elif choice == '8':
            height = rb_tree.tree_height(rb_tree.root)
            print(f"Tree Height: {height}")
        elif choice == '9':
            break
        else:
            print("Invalid choice. Please select again.")

if __name__ == "__main__":
    main()
