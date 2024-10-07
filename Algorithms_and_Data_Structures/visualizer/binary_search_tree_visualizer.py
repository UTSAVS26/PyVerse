import streamlit as st
import graphviz as gv

# TreeNode class definition for Binary Search Tree (BST)
class TreeNode:
    def __init__(self, key):
        self.left = None
        self.right = None
        self.val = key

# Binary Search Tree (BST) class definition
class BinarySearchTree:
    def __init__(self):
        self.root = None

    def insert(self, key):
        if self.root is None:
            self.root = TreeNode(key)
        else:
            self._insert_rec(self.root, key)

    def _insert_rec(self, node, key):
        if key < node.val:
            if node.left is None:
                node.left = TreeNode(key)
            else:
                self._insert_rec(node.left, key)
        else:
            if node.right is None:
                node.right = TreeNode(key)
            else:
                self._insert_rec(node.right, key)

    def delete(self, key):
        self.root = self._delete_rec(self.root, key)

    def _delete_rec(self, node, key):
        if node is None:
            return node

        if key < node.val:
            node.left = self._delete_rec(node.left, key)
        elif key > node.val:
            node.right = self._delete_rec(node.right, key)
        else:
            # Node with only one child or no child
            if node.left is None:
                return node.right
            elif node.right is None:
                return node.left

            # Node with two children, find the inorder successor (smallest in the right subtree)
            temp = self._min_value_node(node.right)
            node.val = temp.val
            node.right = self._delete_rec(node.right, temp.val)

        return node

    def _min_value_node(self, node):
        current = node
        while current.left is not None:
            current = current.left
        return current

    def inorder(self, node):
        result = []
        if node:
            result = self.inorder(node.left)
            result.append(node.val)
            result = result + self.inorder(node.right)
        return result

    def visualize_tree(self):
        dot = gv.Digraph()

        def add_edges(node):
            if node:
                dot.node(str(node.val))
                if node.left:
                    dot.edge(str(node.val), str(node.left.val), label="L")
                    add_edges(node.left)
                if node.right:
                    dot.edge(str(node.val), str(node.right.val), label="R")
                    add_edges(node.right)

        if self.root:
            add_edges(self.root)
        return dot

# Streamlit UI code
def app():
    st.title("Binary Search Tree (BST) Visualization")

    # Initialize Binary Search Tree
    if 'tree' not in st.session_state:
        st.session_state.tree = BinarySearchTree()

    # Insert node input
    insert_value = st.number_input("Insert a value into the BST:", value=0, step=1)
    if st.button("Insert"):
        st.session_state.tree.insert(insert_value)
        st.success(f"Inserted {insert_value} into the BST")

    # Delete node input
    delete_value = st.number_input("Delete a value from the BST:", value=0, step=1)
    if st.button("Delete"):
        st.session_state.tree.delete(delete_value)
        st.success(f"Deleted {delete_value} from the BST")

    # Inorder Traversal display
    if st.button("Inorder Traversal"):
        result = st.session_state.tree.inorder(st.session_state.tree.root)
        st.write("Inorder Traversal:", result)

    # Tree visualization using Graphviz
    if st.button("Visualize Tree"):
        if st.session_state.tree.root:
            dot = st.session_state.tree.visualize_tree()
            st.graphviz_chart(dot.source)
        else:
            st.write("The tree is empty!")

# Run the Streamlit app
if __name__ == "__main__":
    app()
