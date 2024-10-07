import streamlit as st
from graphviz import Digraph

class Node:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None
        self.height = 1  # Every new node starts with height 1

def height(node):
    if node is None:
        return 0
    return node.height

def max(a, b):
    return a if a > b else b

def newNode(data):
    return Node(data)

def getBalance(node):
    if node is None:
        return 0
    return height(node.left) - height(node.right)

# Right rotation for Left-Left case (balance > 1 and data < node.left.data)
def rightRotate(y):
    x = y.left
    T2 = x.right

    # Perform rotation
    x.right = y
    y.left = T2

    # Update heights of the rotated nodes
    y.height = max(height(y.left), height(y.right)) + 1
    x.height = max(height(x.left), height(x.right)) + 1

    return x

# Left rotation for Right-Right case (balance < -1 and data > node.right.data)
def leftRotate(x):
    y = x.right
    T2 = y.left

    # Perform rotation
    y.left = x
    x.right = T2

    # Update heights of the rotated nodes
    x.height = max(height(x.left), height(x.right)) + 1
    y.height = max(height(y.left), height(y.right)) + 1

    return y

# Insert a node and balance the AVL tree
def insert(node, data):
    # Standard BST insertion
    if node is None:
        return newNode(data)

    if data < node.data:
        node.left = insert(node.left, data)
    elif data > node.data:
        node.right = insert(node.right, data)
    else:
        # No duplicates allowed
        return node

    # Update height of the current node after insertion
    node.height = 1 + max(height(node.left), height(node.right))

    # Get balance factor to check for imbalance
    balance = getBalance(node)

    # Left-Left case: single right rotation
    if balance > 1 and data < node.left.data:
        return rightRotate(node)

    # Right-Right case: single left rotation
    if balance < -1 and data > node.right.data:
        return leftRotate(node)

    # Left-Right case: first left-rotate the left child, then right-rotate
    if balance > 1 and data > node.left.data:
        node.left = leftRotate(node.left)
        return rightRotate(node)

    # Right-Left case: first right-rotate the right child, then left-rotate
    if balance < -1 and data < node.right.data:
        node.right = rightRotate(node.right)
        return leftRotate(node)

    # Return the (potentially rotated) root node
    return node

def inOrder(root, ls):
    if root is not None:
        inOrder(root.left, ls)
        ls.append(root.data)
        inOrder(root.right, ls)

def preOrder(root, ls):
    if root is not None:
        ls.append(root.data)
        preOrder(root.left, ls)
        preOrder(root.right, ls)

# Visualizing the tree using Graphviz
def visualize_tree(node, graph=None):
    if graph is None:
        graph = Digraph()  # Create new Digraph object
        graph.attr('node', shape='circle')

    if node is not None:
        # Add the current node to the graph
        graph.node(str(node.data), str(node.data))

        # Recursively add left and right children
        if node.left:
            graph.edge(str(node.data), str(node.left.data))  # Add edge between parent and left child
            visualize_tree(node.left, graph)
        if node.right:
            graph.edge(str(node.data), str(node.right.data))  # Add edge between parent and right child
            visualize_tree(node.right, graph)

    return graph  # Return the graph

def custom_write(ls):
    return ', '.join(f'{item}' for item in ls)

st.title("AVL Tree Visualizer")

# Tabs for file upload and manual entry
tab1, tab2 = st.tabs(["File Upload", "Manual Entry"])

with tab1:
    st.subheader("Upload AVL Tree Data")
    uploaded_file = st.file_uploader("Upload the input file", type="txt")
    
    if uploaded_file is not None:
        content = uploaded_file.read().decode("utf-8")  # Read file content as text
        numbers = [int(num) for num in content.strip().split() if num.isdigit()]  # Extract numbers

        root = None  # Initialize the AVL tree root
        steps = []   # Store each step of the tree construction

        # Insert each number and visualize the tree after each insertion
        for key in numbers:
            root = insert(root, key)
            current_graph = visualize_tree(root, Digraph())  # Capture tree structure at this step
            steps.append(current_graph.source)  # Save graph source

        ino = []  # Store inorder traversal result
        pre = []  # Store preorder traversal result

        inOrder(root, ino)
        preOrder(root, pre)

        st.subheader("Construction of AVL tree:")
        if steps:
            # Slider to navigate through different tree construction steps
            step_index = st.slider('Step', 0, len(steps)-1, 0)
            st.graphviz_chart(steps[step_index])

        st.subheader("Inorder Traversal of the AVL tree:")
        st.write(custom_write(ino))

        st.subheader("Preorder Traversal of the AVL tree:")
        st.write(custom_write(pre))

        st.subheader("Final AVL Tree Structure:")
        final_graph = visualize_tree(root, Digraph())  # Final tree structure
        st.graphviz_chart(final_graph.source)

        # Write inorder traversal to output file
        output_file = 'output.txt'
        try:
            with open(output_file, 'w') as f:
                for item in ino:
                    f.write(str(item)+' ')
        except FileNotFoundError:
            st.error(f"Error: File '{output_file}' does not exist.")

with tab2:
    st.subheader("Enter AVL Tree Data")
    
    # Initialize session state for tree root and numbers
    if 'root' not in st.session_state:
        st.session_state.root = None
        st.session_state.numbers = []

    input_number = st.text_input("Enter a number:")
    if st.button("Add Number"):
        if input_number.isdigit():
            number = int(input_number)
            # Insert the number into the tree and update session state
            st.session_state.root = insert(st.session_state.root, number)
            st.session_state.numbers.append(number)
            st.success(f"Number {number} added.")
        else:
            st.error("Please enter a valid number.")

    # If the tree has been built, show its structure
    if st.session_state.root:
        ino = []  # Inorder traversal result
        pre = []  # Preorder traversal result

        inOrder(st.session_state.root, ino)
        preOrder(st.session_state.root, pre)

        st.subheader("AVL Tree Structure:")
        final_graph = visualize_tree(st.session_state.root, Digraph())  # Display tree structure
        st.graphviz_chart(final_graph.source)

        st.subheader("Inorder Traversal of the AVL tree:")
        st.write(custom_write(ino))

        st.subheader("Preorder Traversal of the AVL tree:")
        st.write(custom_write(pre))
