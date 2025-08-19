import time
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
import random

class FenwickTree:
    """Fenwick Tree (Binary Indexed Tree) for efficient range sum queries and point updates"""
    
    def __init__(self, size: int):
        self.size = size
        self.tree = [0] * (size + 1)  # 1-based indexing
        self.original_array = [0] * size
        self.operation_log = []
    
    def _get_lsb(self, index: int) -> int:
        """Get the least significant bit of index"""
        return index & (-index)
    
    def update(self, index: int, value: int) -> None:
        """Add value to element at index (1-based indexing)"""
        if index < 1 or index > self.size:
            raise ValueError(f"Index {index} out of bounds [1, {self.size}]")
        
        # Update original array
        self.original_array[index - 1] += value
        
        # Update Fenwick Tree
        original_index = index
        while index <= self.size:
            self.tree[index] += value
            index += self._get_lsb(index)
        
        # Log operation
        self.operation_log.append({
            'operation': 'update',
            'index': original_index,
            'value': value,
            'timestamp': time.time()
        })
    def query(self, index: int) -> int:
        """Get sum from index 1 to index (1-based indexing)"""
        if index < 1 or index > self.size:
            raise ValueError(f"Index {index} out of bounds [1, {self.size}]")
        
        result = 0
        while index > 0:
            result += self.tree[index]
            index -= self._get_lsb(index)
        
        return result
    
    def range_query(self, left: int, right: int) -> int:
        """Get sum from left to right inclusive (1-based indexing)"""
        if left < 1 or right > self.size or left > right:
            raise ValueError(f"Invalid range [{left}, {right}]")
        
        if left == 1:
            return self.query(right)
        else:
            return self.query(right) - self.query(left - 1)
    
    def set_value(self, index: int, value: int) -> None:
        """Set value at index (not add)"""
        if index < 1 or index > self.size:
            raise ValueError(f"Index {index} out of bounds [1, {self.size}]")
        
        # Calculate the difference
        current_value = self.original_array[index - 1]
        diff = value - current_value
        
        # Update using the difference
        self.update(index, diff)
    
    def get_array(self) -> List[int]:
        """Get the original array from the tree"""
        return self.original_array.copy()
    
    def range_update(self, left: int, right: int, value: int) -> None:
        """Add value to all elements in range [left, right]"""
        if left < 1 or right > self.size or left > right:
            raise ValueError(f"Invalid range [{left}, {right}]")
        
        # Update each element in the range
        for i in range(left, right + 1):
            self.update(i, value)
    
    def count_inversions(self, arr: List[int]) -> int:
        """Count inversions in an array using coordinate compression"""
        if not arr:
            return 0
        
        # Coordinate compression
        compressed = self.compress_coordinates(arr)
        
        # Create Fenwick Tree for compressed values
        max_val = max(compressed) if compressed else 0
        ft = FenwickTree(max_val)
        
        inversions = 0
        for i in range(len(compressed) - 1, -1, -1):
            # Count elements smaller than current element
            if compressed[i] > 1:
                inversions += ft.query(compressed[i] - 1)
            ft.update(compressed[i], 1)
        
        return inversions
    
    def compress_coordinates(self, arr: List[int]) -> List[int]:
        """Compress coordinates to handle large values efficiently"""
        if not arr:
            return []
        
        # Create mapping of original values to compressed values
        unique_values = sorted(set(arr))
        value_to_index = {val: idx + 1 for idx, val in enumerate(unique_values)}
        
        # Return compressed array
        return [value_to_index[val] for val in arr]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the tree"""
        return {
            'size': self.size,
            'total_operations': len(self.operation_log),
            'current_sum': self.query(self.size),
            'array_sum': sum(self.original_array),
            'max_value': max(self.original_array) if self.original_array else 0,
            'min_value': min(self.original_array) if self.original_array else 0
        }
    
    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze performance metrics"""
        if not self.operation_log:
            return {}
        
        operations = [op['operation'] for op in self.operation_log]
        timestamps = [op['timestamp'] for op in self.operation_log]
        
        # Calculate operation frequencies
        op_counts = defaultdict(int)
        for op in operations:
            op_counts[op] += 1
        
        # Calculate time intervals
        intervals = []
        for i in range(1, len(timestamps)):
            intervals.append(timestamps[i] - timestamps[i-1])
        
        return {
            'total_operations': len(self.operation_log),
            'operation_counts': dict(op_counts),
            'average_interval': np.mean(intervals) if intervals else 0,
            'total_time': timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0
        }

class FenwickTree2D:
    """2D Fenwick Tree for matrix range queries and updates"""
    
    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        self.tree = [[0] * (cols + 1) for _ in range(rows + 1)]
        self.original_matrix = [[0] * cols for _ in range(rows)]
    
    def _get_lsb(self, index: int) -> int:
        """Get the least significant bit of index"""
        return index & (-index)
    
    def update(self, row: int, col: int, value: int) -> None:
        """Add value to element at (row, col) (1-based indexing)"""
        if row < 1 or row > self.rows or col < 1 or col > self.cols:
            raise ValueError(f"Position ({row}, {col}) out of bounds")
        
        # Update original matrix
        self.original_matrix[row - 1][col - 1] += value
        
        # Update 2D Fenwick Tree
        i = row
        while i <= self.rows:
            j = col
            while j <= self.cols:
                self.tree[i][j] += value
                j += self._get_lsb(j)
            i += self._get_lsb(i)
    
    def query(self, row: int, col: int) -> int:
        """Get sum from (1,1) to (row, col) (1-based indexing)"""
        if row < 1 or row > self.rows or col < 1 or col > self.cols:
            raise ValueError(f"Position ({row}, {col}) out of bounds")
        
        result = 0
        i = row
        while i > 0:
            j = col
            while j > 0:
                result += self.tree[i][j]
                j -= self._get_lsb(j)
            i -= self._get_lsb(i)
        
        return result
    
    def range_query(self, row1: int, col1: int, row2: int, col2: int) -> int:
        """Get sum from (row1, col1) to (row2, col2) inclusive"""
        if row1 > row2 or col1 > col2:
            return 0
        
        return (self.query(row2, col2) - 
                self.query(row2, col1 - 1) - 
                self.query(row1 - 1, col2) + 
                self.query(row1 - 1, col1 - 1))
    
    def get_matrix(self) -> List[List[int]]:
        """Get the original matrix from the tree"""
        return [row[:] for row in self.original_matrix]

def build_fenwick_tree(arr: List[int]) -> FenwickTree:
    """Build Fenwick Tree from array"""
    ft = FenwickTree(len(arr))
    for i, value in enumerate(arr):
        ft.update(i + 1, value)
    return ft

def count_inversions(arr: List[int]) -> int:
    """Count inversions using Fenwick Tree"""
    ft = FenwickTree(len(arr))
    return ft.count_inversions(arr)

def analyze_performance(operations: List[Tuple[str, int]]) -> Dict[str, Any]:
    """Analyze performance of Fenwick Tree operations"""
    ft = FenwickTree(1000)
    start_time = time.time()
    
    for op, value in operations:
        if op == 'update':
            ft.update(random.randint(1, 1000), value)
        elif op == 'query':
            ft.query(random.randint(1, 1000))
    
    end_time = time.time()
    
    return {
        'total_operations': len(operations),
        'total_time': end_time - start_time,
        'average_time_per_op': ((end_time - start_time) / len(operations)) if operations else 0.0,
        'tree_statistics': ft.get_statistics()
    }

def generate_test_cases() -> List[Dict[str, Any]]:
    """Generate test cases for Fenwick Tree"""
    test_cases = []
    
    # Test case 1: Basic operations
    ft1 = FenwickTree(5)
    ft1.update(1, 5)
    ft1.update(2, 3)
    ft1.update(3, 7)
    
    test_cases.append({
        'name': 'Basic Operations',
        'tree': ft1,
        'expected_queries': {
            (1, 3): 15,
            (2, 4): 10,
            (1, 1): 5
        }
    })
    
    # Test case 2: Inversion counting
    arr = [3, 1, 4, 2]
    inversions = count_inversions(arr)
    
    test_cases.append({
        'name': 'Inversion Count',
        'array': arr,
        'expected_inversions': 3
    })
    
    # Test case 3: 2D Fenwick Tree
    ft2d = FenwickTree2D(3, 3)
    ft2d.update(1, 1, 5)
    ft2d.update(2, 2, 3)
    
    test_cases.append({
        'name': '2D Fenwick Tree',
        'tree_2d': ft2d,
        'expected_queries': {
            (1, 1, 2, 2): 8,
            (1, 1, 1, 1): 5
        }
    })
    
    return test_cases

def visualize_fenwick_tree(tree: FenwickTree, show_plot: bool = True) -> None:
    """Visualize the Fenwick Tree structure"""
    if tree.size == 0:
        print("Empty tree")
        return
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Tree structure
    G = nx.DiGraph()
    pos = {}
    
    # Add nodes for tree structure
    for i in range(1, tree.size + 1):
        G.add_node(i, value=tree.tree[i])
        pos[i] = (i, 0)
        
        # Add edges based on LSB
        parent = i + tree._get_lsb(i)
        if parent <= tree.size:
            G.add_edge(i, parent)
    
    # Draw tree
    nx.draw(G, pos, ax=ax1, with_labels=True, 
           node_color='lightblue', node_size=1000,
           arrows=True, arrowstyle='->', arrowsize=20)
    
    # Add node labels
    labels = {node: f"{node}\n{tree.tree[node]}" for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, ax=ax1)
    
    ax1.set_title('Fenwick Tree Structure')
    ax1.set_xlim(-1, tree.size + 2)
    ax1.set_ylim(-1, 2)
    
    # Plot 2: Original array vs cumulative sums
    indices = list(range(1, tree.size + 1))
    original_values = tree.original_array
    cumulative_sums = [tree.query(i) for i in indices]
    
    x = np.arange(len(indices))
    width = 0.35
    
    ax2.bar(x - width/2, original_values, width, label='Original Array', alpha=0.7)
    ax2.bar(x + width/2, cumulative_sums, width, label='Cumulative Sums', alpha=0.7)
    
    ax2.set_xlabel('Index')
    ax2.set_ylabel('Value')
    ax2.set_title('Original Array vs Cumulative Sums')
    ax2.set_xticks(x)
    ax2.set_xticklabels(indices)
    ax2.legend()
    
    plt.tight_layout()
    if show_plot:
        plt.show()

def visualize_update_process(tree: FenwickTree, index: int, value: int, show_plot: bool = True) -> None:
    """Visualize the update process for a specific index"""
    if index < 1 or index > tree.size:
        print(f"Invalid index {index}")
        return
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    
    G = nx.DiGraph()
    pos = {}
    
    # Add all nodes
    for i in range(1, tree.size + 1):
        G.add_node(i, value=tree.tree[i])
        pos[i] = (i, 0)
    
    # Add edges and highlight update path
    update_path = set()
    current = index
    while current <= tree.size:
        update_path.add(current)
        parent = current + tree._get_lsb(current)
        if parent <= tree.size:
            G.add_edge(current, parent)
        current = parent
    
    # Draw graph
    node_colors = ['red' if node in update_path else 'lightblue' for node in G.nodes()]
    nx.draw(G, pos, ax=ax, with_labels=True, 
           node_color=node_colors, node_size=1000,
           arrows=True, arrowstyle='->', arrowsize=20)
    
    # Add labels
    labels = {node: f"{node}\n{tree.tree[node]}" for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, ax=ax)
    
    ax.set_title(f'Update Process: Adding {value} to index {index}')
    ax.set_xlim(-1, tree.size + 2)
    ax.set_ylim(-1, 2)
    
    plt.tight_layout()
    if show_plot:
        plt.show()

def visualize_performance_metrics(tree: FenwickTree, show_plot: bool = True) -> None:
    """Visualize performance metrics"""
    if not tree.operation_log:
        print("No operations to analyze")
        return
    
    # Extract data
    operations = [op['operation'] for op in tree.operation_log]
    timestamps = [op['timestamp'] for op in tree.operation_log]
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Operation frequency
    op_counts = defaultdict(int)
    for op in operations:
        op_counts[op] += 1
    
    ax1.bar(op_counts.keys(), op_counts.values(), color='skyblue')
    ax1.set_title('Operation Frequency')
    ax1.set_ylabel('Count')
    
    # Plot 2: Operations over time
    ax2.plot(range(len(timestamps)), timestamps, 'b-', marker='o')
    ax2.set_title('Operations Timeline')
    ax2.set_xlabel('Operation Index')
    ax2.set_ylabel('Timestamp')
    
    # Plot 3: Array values over time
    if tree.original_array:
        ax3.plot(range(1, len(tree.original_array) + 1), tree.original_array, 'g-', marker='s')
        ax3.set_title('Array Values')
        ax3.set_xlabel('Index')
        ax3.set_ylabel('Value')
    
    # Plot 4: Tree statistics
    stats = tree.get_statistics()
    metrics = ['Size', 'Total Ops', 'Current Sum', 'Array Sum']
    values = [stats['size'], stats['total_operations'], 
              stats['current_sum'], stats['array_sum']]
    
    ax4.bar(metrics, values, color=['orange', 'red', 'purple', 'green'])
    ax4.set_title('Tree Statistics')
    ax4.set_ylabel('Value')
    
    plt.tight_layout()
    if show_plot:
        plt.show()

def main():
    """Main function to demonstrate Fenwick Tree"""
    print("=== Fenwick Tree (Binary Indexed Tree) Demo ===\n")
    
    # Create Fenwick Tree
    ft = FenwickTree(10)
    
    print("1. Basic Operations:")
    print("   Inserting elements: 5, 3, 7, 2, 8")
    ft.update(1, 5)
    ft.update(2, 3)
    ft.update(3, 7)
    ft.update(4, 2)
    ft.update(5, 8)
    
    print(f"   Tree size: {ft.size}")
    print(f"   Available indices: 1 to {ft.size}")
    
    print("\n2. Range Queries:")
    sum_1_3 = ft.range_query(1, 3)
    sum_2_4 = ft.range_query(2, 4)
    sum_1_5 = ft.range_query(1, 5)
    
    print(f"   Sum [1,3]: {sum_1_3}")
    print(f"   Sum [2,4]: {sum_2_4}")
    print(f"   Sum [1,5]: {sum_1_5}")
    
    print("\n3. Point Updates:")
    print("   Adding 10 to index 2")
    ft.update(2, 10)
    
    new_sum_1_3 = ft.range_query(1, 3)
    print(f"   New sum [1,3]: {new_sum_1_3}")
    
    print("\n4. Inversion Counting:")
    test_array = [3, 1, 4, 2, 5]
    inversions = count_inversions(test_array)
    print(f"   Array: {test_array}")
    print(f"   Inversions: {inversions}")
    
    print("\n5. 2D Fenwick Tree:")
    ft2d = FenwickTree2D(3, 3)
    ft2d.update(1, 1, 5)
    ft2d.update(2, 2, 3)
    ft2d.update(1, 2, 2)
    
    sum_2d = ft2d.range_query(1, 1, 2, 2)
    print(f"   Sum [1,1] to [2,2]: {sum_2d}")
    
    print("\n6. Performance Analysis:")
    perf = ft.analyze_performance()
    print(f"   Total operations: {perf.get('total_operations', 0)}")
    print(f"   Operation counts: {perf.get('operation_counts', {})}")
    
    print("\n7. Tree Statistics:")
    stats = ft.get_statistics()
    print(f"   Current sum: {stats['current_sum']}")
    print(f"   Array sum: {stats['array_sum']}")
    print(f"   Max value: {stats['max_value']}")
    print(f"   Min value: {stats['min_value']}")
    
    print("\n8. Visualization:")
    print("   Generating visualizations...")
    
    # Generate visualizations
    visualize_fenwick_tree(ft, show_plot=False)
    visualize_update_process(ft, 2, 10, show_plot=False)
    visualize_performance_metrics(ft, show_plot=False)
    
    print("   Visualizations created successfully!")
    
    print("\n9. Test Cases:")
    test_cases = generate_test_cases()
    print(f"   Generated {len(test_cases)} test cases")
    
    for i, test_case in enumerate(test_cases):
        print(f"   Test case {i+1}: {test_case['name']}")
    
    print("\n=== Demo Complete ===")
    
    # Show one visualization
    visualize_fenwick_tree(ft)

if __name__ == "__main__":
    main() 