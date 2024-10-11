import streamlit as st
import graphviz as gv
from collections import deque
  
# TreeNode class definition for Binary Tree
class TreeNode:
    def __init__(self, key):
        self.left = None
        self.right = None
        self.val = key

# Binary Tree class definition
class BinaryTree:
    def __init__(self):
        self.root = None

    def insert(self, key):
        new_node = TreeNode(key)
        if self.root is None:
            self.root = new_node
        else:
            # Level-order traversal to insert at the first available position
            q = deque([self.root])
            while q:
                current = q.popleft()
                if not current.left:
                    current.left = new_node
                    break
                else:
                    q.append(current.left)
                if not current.right:
                    current.right = new_node
                    break
                else:
                    q.append(current.right)

    def delete(self, key):
        if self.root is None:
            return None

        if self.root.left is None and self.root.right is None:
            if self.root.val == key:
                self.root = None
            return self.root

        # Level-order traversal to find the node to delete
        q = deque([self.root])
        node_to_delete = None
        last_node = None
        while q:
            last_node = q.popleft()
            if last_node.val == key:
                node_to_delete = last_node
            if last_node.left:
                q.append(last_node.left)
            if last_node.right:
                q.append(last_node.right)

        # If node_to_delete is found, replace it with the last node and remove last node
        if node_to_delete:
            node_to_delete.val = last_node.val
            self._delete_last_node(self.root, last_node)

    def _delete_last_node(self, root, last_node):
        # Level-order traversal to remove the last node
        q = deque([root])
        while q:
            current = q.popleft()
            if current.left:
                if current.left == last_node:
                    current.left = None
                    return
                q.append(current.left)
            if current.right:
                if current.right == last_node:
                    current.right = None
                    return
                q.append(current.right)

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
    st.title("Binary Tree Visualization")

    # Create a BinaryTree instance
    if 'tree' not in st.session_state:
        st.session_state.tree = BinaryTree()

    # Insert node input
    insert_value = st.number_input("Insert a value into the Binary Tree:", value=0, step=1)
    if st.button("Insert"):
        st.session_state.tree.insert(insert_value)
        st.success(f"Inserted {insert_value} into the tree")

    # Delete node input
    delete_value = st.number_input("Delete a value from the Binary Tree:", value=0, step=1)
    if st.button("Delete"):
        st.session_state.tree.delete(delete_value)
        st.success(f"Deleted {delete_value} from the tree")

    # Traversals
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
