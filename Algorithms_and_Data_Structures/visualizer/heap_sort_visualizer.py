import streamlit as st
import graphviz
import time
import numpy as np
     
# Node class for Binary Search Tree (BST)
class BSTNode:
    def __init__(self, key):
        self.left = None
        self.right = None
        self.value = key

# Insert function for Binary Search Tree
def insert(root, key):
    if root is None:
        return BSTNode(key)
    else:
        if key < root.value:
            root.left = insert(root.left, key)
        else:
            root.right = insert(root.right, key)
    return root

# Inorder Traversal
def inorder(root, steps):
    if root:
        inorder(root.left, steps)
        steps.append(root.value)
        inorder(root.right, steps)

# Preorder Traversal
def preorder(root, steps):
    if root:
        steps.append(root.value)
        preorder(root.left, steps)
        preorder(root.right, steps)

# Postorder Traversal
def postorder(root, steps):
    if root:
        postorder(root.left, steps)
        postorder(root.right, steps)
        steps.append(root.value)

# Visualize the Binary Search Tree using graphviz
def visualize_bst(root, dot=None, parent=None, direction=None):
    if dot is None:
        dot = graphviz.Digraph()
    if root:
        node_label = str(root.value)
        dot.node(node_label)
        if parent:
            if direction == 'left':
                dot.edge(str(parent), node_label, label='L')
            elif direction == 'right':
                dot.edge(str(parent), node_label, label='R')
        visualize_bst(root.left, dot, root.value, 'left')
        visualize_bst(root.right, dot, root.value, 'right')
    return dot

def main():
    st.title("Binary Search Tree (BST) Visualizer")

    # Input options
    st.sidebar.title("Options")
    
    # File Upload
    uploaded_file = st.sidebar.file_uploader("Upload a text file containing an array", type=["txt"])
    if uploaded_file is not None:
        content = uploaded_file.read().decode("utf-8")
        arr = list(map(int, content.split()))
        st.session_state.arr = arr
        st.write("### Uploaded Array")
        st.write(arr)

    # Random Number Generator
    num_inputs = st.sidebar.number_input("Number of Random Inputs", min_value=1, value=10)
    if st.sidebar.button("Generate Random Array"):
        arr = np.random.randint(1, 100, size=num_inputs).tolist()
        st.session_state.arr = arr
        st.write("### Generated Random Array")
        st.write(arr)

    # Insert Element
    if 'arr' in st.session_state:
        new_element = st.sidebar.number_input("Element to Insert", value=0)
        if st.sidebar.button("Insert Element"):
            st.session_state.arr.append(new_element)
            st.write("### Array after Insertion")
            st.write(st.session_state.arr)

    if 'arr' in st.session_state:
        arr = st.session_state.arr

        # Build the BST
        bst_root = None
        for value in arr:
            bst_root = insert(bst_root, value)

        st.write("## Binary Search Tree Visualization")
        dot = visualize_bst(bst_root)
        st.graphviz_chart(dot)

        # BST Traversal options
        traversal_method = st.selectbox("Select Traversal Method", ["Inorder", "Preorder", "Postorder"])

        traversal_steps = []
        if traversal_method == "Inorder":
            inorder(bst_root, traversal_steps)
        elif traversal_method == "Preorder":
            preorder(bst_root, traversal_steps)
        elif traversal_method == "Postorder":
            postorder(bst_root, traversal_steps)

        st.write(f"### {traversal_method} Traversal:")
        st.write(traversal_steps)

if __name__ == "__main__":
    main()
