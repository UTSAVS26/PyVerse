import time
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
import matplotlib.pyplot as plt
import networkx as nx
from collections import deque

class CartesianNode:
    """Node for Cartesian Tree"""
    
    def __init__(self, value: int, index: int):
        self.value = value
        self.index = index
        self.left = None
        self.right = None
        self.parent = None

class CartesianTree:
    """Cartesian Tree for efficient range minimum queries"""
    
    def __init__(self, array: List[int]):
        self.array = array
        self.root = None
        self.n = len(array)
        self.sparse_table = None
        self.log_table = None
        self.sparse_table_max = None  # for range_max queries
        self.index_to_node: Dict[int, CartesianNode] = {}
        self._build_tree()
        self._build_sparse_table()
    def _build_tree(self) -> None:
        """Build Cartesian tree using stack-based algorithm"""
        if not self.array:
            return

        stack: list[CartesianNode] = []
        for i, value in enumerate(self.array):
            node = CartesianNode(value, i)
            last_popped = None

            # Pop while maintaining increasing stack (min-heap Cartesian tree)
            while stack and stack[-1].value > value:
                last_popped = stack.pop()

            # Link current node as right child of new stack top (if any)
            if stack:
                stack[-1].right = node
                node.parent = stack[-1]
            else:
                self.root = node

            # Attach last popped chain as left child of current node
            if last_popped:
                node.left = last_popped
                last_popped.parent = node

            # Index map for O(1) lookups
            if getattr(self, "index_to_node", None) is not None:
                self.index_to_node[i] = node

            stack.append(node)
    def _build_sparse_table(self) -> None:
        """Build sparse table for O(1) range queries"""
        if not self.array:
            return

        n = len(self.array)
        log_n = int(np.log2(n)) + 1

        # Initialize sparse tables
        self.sparse_table     = [[0] * log_n for _ in range(n)]  # min indices
        self.sparse_table_max = [[0] * log_n for _ in range(n)]  # max indices
        self.log_table        = [0] * (n + 1)

        # Fill log table
        for i in range(2, n + 1):
            self.log_table[i] = self.log_table[i // 2] + 1

        # Base case for both tables
        for i in range(n):
            self.sparse_table[i][0]     = i
            self.sparse_table_max[i][0] = i

        # Build both min and max tables
        for j in range(1, log_n):
            span = 1 << (j - 1)
            for i in range(n - (1 << j) + 1):
                # Min table
                left_min  = self.sparse_table[i][j - 1]
                right_min = self.sparse_table[i + span][j - 1]
                self.sparse_table[i][j] = (
                    left_min if self.array[left_min] <= self.array[right_min]
                    else right_min
                )
                # Max table
                left_max  = self.sparse_table_max[i][j - 1]
                right_max = self.sparse_table_max[i + span][j - 1]
                self.sparse_table_max[i][j] = (
                    left_max if self.array[left_max] >= self.array[right_max]
                    else right_max
                )
    def range_min_query(self, left: int, right: int) -> int:
        """Find minimum value in range [left, right]"""
        if left < 0 or right >= self.n or left > right:
            raise ValueError(f"Invalid range [{left}, {right}]")
        
        if left == right:
            return self.array[left]
        
        length = right - left + 1
        k = self.log_table[length]
        
        left_idx = self.sparse_table[left][k]
        right_idx = self.sparse_table[right - (1 << k) + 1][k]
        
        return min(self.array[left_idx], self.array[right_idx])
    
    def range_max_query(self, left: int, right: int) -> int:
        """Find maximum value in range [left, right]"""
        if left < 0 or right >= self.n or left > right:
            raise ValueError(f"Invalid range [{left}, {right}]")
        
        if left == right:
            return self.array[left]
        
        length = right - left + 1
        k = self.log_table[length]
        
        left_idx = self.sparse_table_max[left][k]
        right_idx = self.sparse_table_max[right - (1 << k) + 1][k]
        return max(self.array[left_idx], self.array[right_idx])
    def get_lca(self, i: int, j: int) -> Optional[CartesianNode]:
        """Find lowest common ancestor of nodes i and j"""
        if i < 0 or i >= self.n or j < 0 or j >= self.n:
            return None
        
        # Find nodes in O(1) from index map
        node_i = self.index_to_node.get(i)
        node_j = self.index_to_node.get(j)
        
        if not node_i or not node_j:
            return None
        
        return self._find_lca(node_i, node_j)
    
    def _find_node_by_index(self, node: Optional[CartesianNode], index: int) -> Optional[CartesianNode]:
        """Find node by array index"""
        if not node:
            return None
        
        if node.index == index:
            return node
        
        left_result = self._find_node_by_index(node.left, index)
        if left_result:
            return left_result
        
        return self._find_node_by_index(node.right, index)
    
    def _find_lca(self, node1: CartesianNode, node2: CartesianNode) -> CartesianNode:
        """Find lowest common ancestor of two nodes"""
        # Get paths to root
        path1 = self._get_path_to_root(node1)
        path2 = self._get_path_to_root(node2)
        
        # Find LCA
        i = len(path1) - 1
        j = len(path2) - 1
        
        lca = None
        while i >= 0 and j >= 0 and path1[i] == path2[j]:
            lca = path1[i]
            i -= 1
            j -= 1
        
        return lca
    
    def _get_path_to_root(self, node: CartesianNode) -> List[CartesianNode]:
        """Get path from node to root"""
        path = []
        current = node
        while current:
            path.append(current)
            current = current.parent
        return path
    
    def inorder_traversal(self) -> List[int]:
        """Get inorder traversal of tree"""
        result = []
        
        def _inorder(node: Optional[CartesianNode]):
            if node:
                _inorder(node.left)
                result.append(node.value)
                _inorder(node.right)
        
        _inorder(self.root)
        return result
    
    def get_height(self) -> int:
        """Get height of tree"""
        def _height(node: Optional[CartesianNode]) -> int:
            if not node:
                return 0
            return 1 + max(_height(node.left), _height(node.right))
        
        return _height(self.root)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the tree"""
        return {
            'size': self.n,
            'height': self.get_height(),
            'min_value': min(self.array) if self.array else None,
            'max_value': max(self.array) if self.array else None,
            'array_sum': sum(self.array) if self.array else 0,
            'has_sparse_table': self.sparse_table is not None
        }
    
    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze performance metrics"""
        if not self.array:
            return {}
        
        # Test range queries
        start_time = time.time()
        for _ in range(1000):
            left = np.random.randint(0, self.n)
            right = np.random.randint(left, self.n)
            self.range_min_query(left, right)
        
        query_time = time.time() - start_time
        
        return {
            'array_size': self.n,
            'tree_height': self.get_height(),
            'query_time_1000_ops': query_time,
            'average_query_time': query_time / 1000,
            'statistics': self.get_statistics()
        }

def build_cartesian_tree(array: List[int]) -> CartesianTree:
    """Build Cartesian tree from array"""
    return CartesianTree(array)

def range_minimum_query(array: List[int], left: int, right: int) -> int:
    """Perform range minimum query on array"""
    ct = CartesianTree(array)
    return ct.range_min_query(left, right)

def generate_test_cases() -> List[Dict[str, Any]]:
    """Generate test cases for Cartesian Tree"""
    test_cases = []
    
    # Test case 1: Basic operations
    arr1 = [3, 1, 4, 1, 5, 9, 2, 6]
    ct1 = CartesianTree(arr1)
    
    test_cases.append({
        'name': 'Basic Operations',
        'tree': ct1,
        'array': arr1,
        'expected_queries': {
            (2, 5): 1,  # min in range [2,5]
            (1, 4): 1,  # min in range [1,4]
            (0, 7): 1   # min in entire array
        }
    })
    
    # Test case 2: Edge cases
    arr2 = [1]
    ct2 = CartesianTree(arr2)
    
    test_cases.append({
        'name': 'Single Element',
        'tree': ct2,
        'array': arr2,
        'expected_queries': {
            (0, 0): 1
        }
    })
    
    # Test case 3: Duplicate elements
    arr3 = [1, 1, 1, 1]
    ct3 = CartesianTree(arr3)
    
    test_cases.append({
        'name': 'Duplicate Elements',
        'tree': ct3,
        'array': arr3,
        'expected_queries': {
            (0, 3): 1,
            (1, 2): 1
        }
    })
    
    return test_cases

def visualize_cartesian_tree(tree: CartesianTree, show_plot: bool = True) -> None:
    """Visualize the Cartesian tree structure"""
    if not tree.root:
        print("Empty tree")
        return
    
    # Create networkx graph
    G = nx.DiGraph()
    pos = {}
    
    def add_nodes(node: Optional[CartesianNode], x: float = 0, y: float = 0, 
                 level: int = 0, width: float = 2.0):
        if node is None:
            return
        
        G.add_node(id(node), value=node.value, index=node.index)
        pos[id(node)] = (x, -y)
        
        if node.left:
            G.add_edge(id(node), id(node.left))
            add_nodes(node.left, x - width/2, y + 1, level + 1, width/2)
        
        if node.right:
            G.add_edge(id(node), id(node.right))
            add_nodes(node.right, x + width/2, y + 1, level + 1, width/2)
    
    add_nodes(tree.root)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Tree structure
    nx.draw(G, pos, ax=ax1, with_labels=True, 
           node_color='lightblue', node_size=1000,
           arrows=True, arrowstyle='->', arrowsize=20)
    
    labels = {node: f"{G.nodes[node]['value']}\n({G.nodes[node]['index']})" 
              for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, ax=ax1)
    
    ax1.set_title('Cartesian Tree Structure (Value/Index)')
    
    # Plot 2: Original array vs tree values
    indices = list(range(len(tree.array)))
    array_values = tree.array
    tree_values = tree.inorder_traversal()
    
    x = np.arange(len(indices))
    width = 0.35
    
    ax2.bar(x - width/2, array_values, width, label='Original Array', alpha=0.7)
    ax2.bar(x + width/2, tree_values, width, label='Tree Inorder', alpha=0.7)
    ax2.set_xlabel('Index')
    ax2.set_ylabel('Value')
    ax2.set_title('Original Array vs Tree Inorder')
    ax2.set_xticks(x)
    ax2.set_xticklabels(indices)
    ax2.legend()
    
    plt.tight_layout()
    if show_plot:
        plt.show()

def visualize_query_process(tree: CartesianTree, left: int, right: int, 
                          query_type: str = 'min', show_plot: bool = True) -> None:
    """Visualize the query process"""
    if left < 0 or right >= tree.n or left > right:
        print(f"Invalid range [{left}, {right}]")
        return
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot array with highlighted range
    indices = list(range(len(tree.array)))
    values = tree.array
    
    # Color the range
    colors = ['red' if left <= i <= right else 'lightblue' for i in range(len(values))]
    
    bars = ax.bar(indices, values, color=colors, alpha=0.7)
    
    # Highlight the result
    if query_type == 'min':
        result = tree.range_min_query(left, right)
        result_indices = [i for i in range(left, right + 1) if tree.array[i] == result]
    else:
        result = tree.range_max_query(left, right)
        result_indices = [i for i in range(left, right + 1) if tree.array[i] == result]
    
    for idx in result_indices:
        bars[idx].set_color('green')
        bars[idx].set_alpha(0.9)
    
    ax.set_xlabel('Index')
    ax.set_ylabel('Value')
    ax.set_title(f'Range {query_type.upper()} Query: [{left}, {right}] = {result}')
    ax.set_xticks(indices)
    ax.set_xticklabels(indices)
    
    # Add text annotations
    for i, v in enumerate(values):
        ax.text(i, v + 0.1, str(v), ha='center', va='bottom')
    
    plt.tight_layout()
    if show_plot:
        plt.show()

def visualize_performance_metrics(tree: CartesianTree, show_plot: bool = True) -> None:
    """Visualize performance metrics"""
    if not tree.array:
        print("Empty tree")
        return
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Array values
    indices = list(range(len(tree.array)))
    values = tree.array
    
    ax1.bar(indices, values, color='skyblue', alpha=0.7)
    ax1.set_title('Original Array')
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Value')
    
    # Plot 2: Tree height vs array size
    stats = tree.get_statistics()
    metrics = ['Size', 'Height', 'Min Value', 'Max Value']
    stat_values = [stats['size'], stats['height'], 
                   stats['min_value'] or 0, stats['max_value'] or 0]
    
    ax2.bar(metrics, stat_values, color=['orange', 'red', 'green', 'purple'])
    ax2.set_title('Tree Statistics')
    ax2.set_ylabel('Value')
    
    # Plot 3: Performance analysis
    perf = tree.analyze_performance()
    if perf:
        perf_metrics = ['Array Size', 'Tree Height', 'Query Time (ms)']
        perf_values = [perf['array_size'], perf['tree_height'], 
                      perf['average_query_time'] * 1000]
        
        ax3.bar(perf_metrics, perf_values, color=['blue', 'green', 'red'])
        ax3.set_title('Performance Metrics')
        ax3.set_ylabel('Value')
    
    # Plot 4: Sparse table visualization (if available)
    if tree.sparse_table:
        sparse_data = np.array(tree.sparse_table)
        im = ax4.imshow(sparse_data, cmap='viridis', aspect='auto')
        ax4.set_title('Sparse Table')
        ax4.set_xlabel('Log Level')
        ax4.set_ylabel('Array Index')
        plt.colorbar(im, ax=ax4)
    
    plt.tight_layout()
    if show_plot:
        plt.show()

def main():
    """Main function to demonstrate Cartesian Tree"""
    print("=== Cartesian Tree Demo ===\n")
    
    # Create Cartesian tree
    arr = [3, 1, 4, 1, 5, 9, 2, 6]
    ct = CartesianTree(arr)
    
    print("1. Basic Operations:")
    print(f"   Array: {arr}")
    print(f"   Tree size: {ct.n}")
    print(f"   Tree height: {ct.get_height()}")
    
    print("\n2. Range Minimum Queries:")
    queries = [(2, 5), (1, 4), (0, 7)]
    for left, right in queries:
        result = ct.range_min_query(left, right)
        print(f"   RMQ[{left}, {right}] = {result}")
    
    print("\n3. Range Maximum Queries:")
    for left, right in queries:
        result = ct.range_max_query(left, right)
        print(f"   RMQ[{left}, {right}] = {result}")
    
    print("\n4. LCA Queries:")
    lca_pairs = [(2, 6), (1, 4), (0, 7)]
    for i, j in lca_pairs:
        lca = ct.get_lca(i, j)
        if lca:
            print(f"   LCA({i}, {j}) = {lca.value} at index {lca.index}")
        else:
            print(f"   LCA({i}, {j}) = None")
    
    print("\n5. Tree Traversal:")
    inorder = ct.inorder_traversal()
    print(f"   Inorder traversal: {inorder}")
    
    print("\n6. Performance Analysis:")
    perf = ct.analyze_performance()
    print(f"   Array size: {perf.get('array_size', 0)}")
    print(f"   Tree height: {perf.get('tree_height', 0)}")
    print(f"   Average query time: {perf.get('average_query_time', 0):.6f} seconds")
    
    print("\n7. Tree Statistics:")
    stats = ct.get_statistics()
    print(f"   Size: {stats['size']}")
    print(f"   Height: {stats['height']}")
    print(f"   Min value: {stats['min_value']}")
    print(f"   Max value: {stats['max_value']}")
    print(f"   Has sparse table: {stats['has_sparse_table']}")
    
    print("\n8. Visualization:")
    print("   Generating visualizations...")
    
    # Generate visualizations
    visualize_cartesian_tree(ct, show_plot=False)
    visualize_query_process(ct, 2, 5, 'min', show_plot=False)
    visualize_performance_metrics(ct, show_plot=False)
    
    print("   Visualizations created successfully!")
    
    print("\n9. Test Cases:")
    test_cases = generate_test_cases()
    print(f"   Generated {len(test_cases)} test cases")
    
    for i, test_case in enumerate(test_cases):
        print(f"   Test case {i+1}: {test_case['name']}")
    
    print("\n=== Demo Complete ===")
    
    # Show one visualization
    visualize_cartesian_tree(ct)

if __name__ == "__main__":
    main() 