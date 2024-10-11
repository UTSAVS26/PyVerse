import streamlit as st
import graphviz as gv
    
# TreeNode class definition
class TreeNode:
    def __init__(self, key):
        self.left = None
        self.right = None
        self.val = key

# BinaryTree class definition
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
        result = []
        if node:
            result = self.inorder(node.left)
            result.append(node.val)
            result = result + self.inorder(node.right)
        return result

    def preorder(self, node):
        result = []
        if node:
            result.append(node.val)
            result = result + self.preorder(node.left)
            result = result + self.preorder(node.right)
        return result

    def postorder(self, node):
        result = []
        if node:
            result = self.postorder(node.left)
            result = result + self.postorder(node.right)
            result.append(node.val)
        return result

    def visualize_tree(self):
        # Use Graphviz to create a visual representation of the tree
        dot = gv.Digraph()

        def add_edges(node):
            if node:
                # Add the node itself to the graph
                dot.node(str(node.val))
                # If there's a left child, add the edge and recursively process the left subtree
                if node.left:
                    dot.edge(str(node.val), str(node.left.val), label="L")
                    add_edges(node.left)
                # If there's a right child, add the edge and recursively process the right subtree
                if node.right:
                    dot.edge(str(node.val), str(node.right.val), label="R")
                    add_edges(node.right)

        if self.root:
            add_edges(self.root)  # Start adding edges from the root
        return dot

# Streamlit UI code
def app():
    st.title("Binary Tree Visualization")

    # Create a BinaryTree instance
    if 'tree' not in st.session_state:
        st.session_state.tree = BinaryTree()

    # Insert node input
    insert_value = st.number_input("Insert a value into the tree:", value=0, step=1)
    if st.button("Insert"):
        st.session_state.tree.insert(insert_value)
        st.success(f"Inserted {insert_value} into the tree")

    # Traversals
    if st.button("Inorder Traversal"):
        result = st.session_state.tree.inorder(st.session_state.tree.root)
        st.write("Inorder Traversal:", result)

    if st.button("Preorder Traversal"):
        result = st.session_state.tree.preorder(st.session_state.tree.root)
        st.write("Preorder Traversal:", result)

    if st.button("Postorder Traversal"):
        result = st.session_state.tree.postorder(st.session_state.tree.root)
        st.write("Postorder Traversal:", result)

    # Tree visualization using graphviz
    if st.button("Visualize Tree"):
        if st.session_state.tree.root:
            dot = st.session_state.tree.visualize_tree()
            st.graphviz_chart(dot.source)
        else:
            st.write("Tree is empty!")

# Run the Streamlit app
if __name__ == "__main__":
    app()
