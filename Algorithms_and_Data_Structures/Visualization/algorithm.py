import streamlit as st
import numpy as np
import graphviz
import time
import base64

# AVL Tree classes and visualization
class AVLNode:
    def __init__(self, key):
        self.key = key
        self.height = 1
        self.left = None
        self.right = None

class AVLTree:
    def insert(self, root, key):
        if not root:
            return AVLNode(key)
        elif key < root.key:
            root.left = self.insert(root.left, key)
        else:
            root.right = self.insert(root.right, key)

        root.height = 1 + max(self.get_height(root.left), self.get_height(root.right))
        balance = self.get_balance(root)

        # Left Left
        if balance > 1 and key < root.left.key:
            return self.right_rotate(root)
        # Right Right
        if balance < -1 and key > root.right.key:
            return self.left_rotate(root)
        # Left Right
        if balance > 1 and key > root.left.key:
            root.left = self.left_rotate(root.left)
            return self.right_rotate(root)
        # Right Left
        if balance < -1 and key < root.right.key:
            root.right = self.right_rotate(root.right)
            return self.left_rotate(root)

        return root

    def left_rotate(self, z):
        y = z.right
        T2 = y.left
        y.left = z
        z.right = T2
        z.height = 1 + max(self.get_height(z.left), self.get_height(z.right))
        y.height = 1 + max(self.get_height(y.left), self.get_height(y.right))
        return y

    def right_rotate(self, z):
        y = z.left
        T3 = y.right
        y.right = z
        z.left = T3
        z.height = 1 + max(self.get_height(z.left), self.get_height(z.right))
        y.height = 1 + max(self.get_height(y.left), self.get_height(y.right))
        return y

    def get_height(self, root):
        if not root:
            return 0
        return root.height

    def get_balance(self, root):
        if not root:
            return 0
        return self.get_height(root.left) - self.get_height(root.right)

def visualize_avl(root, dot=None):
    if dot is None:
        dot = graphviz.Digraph()
    if root:
        dot.node(str(root.key), str(root.key))
        if root.left:
            dot.edge(str(root.key), str(root.left.key))
            visualize_avl(root.left, dot)
        if root.right:
            dot.edge(str(root.key), str(root.right.key))
            visualize_avl(root.right, dot)
    return dot

# Graph Coloring visualization with forced color usage
def graph_coloring(graph, num_colors):
    coloring = [-1] * len(graph)  # -1 means no color assigned yet
    coloring[0] = 0  # Assign the first color to the first vertex
    available = [True] * num_colors  # Available colors array

    for u in range(1, len(graph)):
        for i in graph[u]:
            if coloring[i] != -1:
                available[coloring[i]] = False

        color = 0
        while color < num_colors and not available[color]:
            color += 1

        coloring[u] = min(color, num_colors - 1)
        available = [True] * num_colors  # Reset for the next vertex

    if len(set(coloring)) < num_colors:
        for i in range(num_colors - len(set(coloring))):
            node_to_recolor = i % len(coloring)
            coloring[node_to_recolor] = i

    return coloring

def visualize_coloring(graph, coloring):
    dot = graphviz.Graph()
    colors = ['#FFDDC1', '#FFABAB', '#FFC3A0', '#FF677D', '#D4A5A5', '#333333']  # Light colors

    for node in range(len(graph)):
        color = colors[coloring[node] % len(colors)]  # Use light colors in a loop
        dot.node(str(node), str(node), style='filled', fillcolor=color)
        for neighbor in graph[node]:
            if node < neighbor:
                dot.edge(str(node), str(neighbor))
    return dot

# Styles and other heap sort-related functions
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

def set_app_style():
    image_path = r"C:\Users\Greeshma G\Pictures\Saved Pictures\pic.jpg"
    st.markdown(
        f"""
        <style>
            .stApp {{
                background-image: linear-gradient(rgba(255, 255, 255, 0.5), rgba(255, 255, 255, 0.5)), url("data:image/png;base64,{image_to_base64(image_path)}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                color: #000000; /* Make all text dark black */
            }}
            .stSidebarContent {{
                background: rgba(255, 255, 255, 0.8); /* Transparent white background for the sidebar */
                color: #000000; /* Sidebar text color */
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# Heap Sort Functions
def sift_up(arr, i, steps, sorted_indices):
    parent = (i - 1) // 2
    while i > 0 and arr[parent] < arr[i]:
        steps.append((arr.copy(), set(sorted_indices), i, "red"))
        arr[i], arr[parent] = arr[parent], arr[i]
        i = parent
        parent = (i - 1) // 2
    steps.append((arr.copy(), set(sorted_indices), i, "green"))

def topDownHeapSort(arr):
    n = len(arr)
    steps = [(arr.copy(), set(), -1, "")]
    sorted_indices = set()
    for i in range(n):
        sift_up(arr, i, steps, sorted_indices)
    for i in range(n-1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        sorted_indices.add(i)
        steps.append((arr.copy(), set(sorted_indices), i, "green"))
        heapify(arr, i, 0, steps, sorted_indices)
    return steps

def sift_down(arr, n, i, steps, sorted_indices):
    largest = i
    l = 2 * i + 1
    r = 2 * i + 2
    if l < n and arr[l] > arr[largest]:
        largest = l
    if r < n and arr[r] > arr[largest]:
        largest = r
    if largest != i:
        steps.append((arr.copy(), set(sorted_indices), i, "red", largest))
        arr[i], arr[largest] = arr[largest], arr[i]
        sift_down(arr, n, largest, steps, sorted_indices)

def bottomUpHeapSort(arr):
    n = len(arr)
    steps = [(arr.copy(), set(), -1, "", -1)]
    sorted_indices = set()
    for i in range(n // 2 - 1, -1, -1):
        sift_down(arr, n, i, steps, sorted_indices)
    for i in range(n-1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        sorted_indices.add(i)
        steps.append((arr.copy(), set(sorted_indices), 0, "green", i))
        sift_down(arr, i, 0, steps, sorted_indices)
    return steps

def heapify(arr, n, i, steps, sorted_indices):
    largest = i
    l = 2 * i + 1
    r = 2 * i + 2
    if l < n and arr[i] < arr[l]:
        largest = l
    if r < n and arr[largest] < arr[r]:
        largest = r
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        steps.append((arr.copy(), set(sorted_indices), i, "red"))
        heapify(arr, n, largest, steps, sorted_indices)

def visualize_heap(arr, sorted_indices, current_index, current_color, swap_with=None):
    dot = graphviz.Digraph()
    n = len(arr)
    for i in range(n):
        color = "green" if i in sorted_indices else "black"
        label = str(arr[i])
        if i == current_index or i == swap_with:
            color = current_color
        dot.node(str(i), label, color=color)
        if 2 * i + 1 < n and 2 * i + 1 not in sorted_indices:
            dot.edge(str(i), str(2 * i + 1), color="black")
        if 2 * i + 2 not in sorted_indices and 2 * i + 2 < n:
            dot.edge(str(i), str(2 * i + 2), color="black")
    return dot

def highlight_array(arr, sorted_indices):
    highlighted_arr = []
    for i in range(len(arr)):
        if i in sorted_indices:
            highlighted_arr.append(f"<span style='color:green'>{arr[i]}</span>")
        else:
            highlighted_arr.append(str(arr[i]))
    return highlighted_arr

def main():
    set_app_style()
    st.title("Algorithm Visualizers")
    
    algorithm = st.sidebar.selectbox("Select Algorithm", ["Heap Sort", "AVL Tree", "Graph Coloring"])

    if algorithm == "Heap Sort":
        st.subheader("Heap Sort Visualization")
        input_data = st.text_input("Enter numbers separated by commas:", "3, 1, 4, 1, 5, 9, 2, 6, 5, 3")
        if st.button("Visualize"):
            arr = list(map(int, input_data.split(",")))
            steps = topDownHeapSort(arr)
            for step in steps:
                arr, sorted_indices, current_index, current_color = step[:4]
                swap_with = step[4] if len(step) > 4 else None
                dot = visualize_heap(arr, sorted_indices, current_index, current_color, swap_with)
                st.graphviz_chart(dot)
                time.sleep(0.5)

    elif algorithm == "AVL Tree":
        st.subheader("AVL Tree Visualization")
        numbers = st.text_input("Enter numbers to insert into AVL Tree, separated by commas:", "10, 20, 30, 40, 50, 25")
        if st.button("Visualize"):
            nums = list(map(int, numbers.split(",")))
            root = None
            for num in nums:
                root = AVLTree().insert(root, num)
            dot = visualize_avl(root)
            st.graphviz_chart(dot)

    elif algorithm == "Graph Coloring":
        st.subheader("Graph Coloring Visualization")
        graph_input = st.text_input("Enter the graph as an adjacency list (e.g., 0: 1, 2; 1: 0, 2; 2: 0, 1):")
        num_colors = st.number_input("Enter the number of colors:", min_value=1, value=3)
        if st.button("Visualize"):
            graph = [list(map(int, node.split(":")[1].split(","))) for node in graph_input.split(";") if node]
            coloring = graph_coloring(graph, num_colors)
            dot = visualize_coloring(graph, coloring)
            st.graphviz_chart(dot)

if __name__ == "__main__":
    main()
