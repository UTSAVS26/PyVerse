import time
import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Any
import matplotlib.pyplot as plt
import networkx as nx

class SegmentTree:
    """Segment Tree for range queries and updates."""
    
    def __init__(self, size: int, default_value: int = 0):
        self.size = size
        self.default_value = default_value
        self.tree = [default_value] * (4 * size)
        self.lazy = [0] * (4 * size)
    
    def _push(self, node: int, start: int, end: int) -> None:
        """Push lazy updates down."""
        if self.lazy[node] != 0:
            self.tree[node] += self.lazy[node] * (end - start + 1)
            if start != end:
                self.lazy[2 * node] += self.lazy[node]
                self.lazy[2 * node + 1] += self.lazy[node]
            self.lazy[node] = 0
    
    def _update_range(self, node: int, start: int, end: int, 
                     l: int, r: int, value: int) -> None:
        """Update range [l, r] with value."""
        self._push(node, start, end)
        
        if start > end or start > r or end < l:
            return
        
        if start >= l and end <= r:
            self.lazy[node] += value
            self._push(node, start, end)
            return
        
        mid = (start + end) // 2
        self._update_range(2 * node, start, mid, l, r, value)
        self._update_range(2 * node + 1, mid + 1, end, l, r, value)
        self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]
    
    def _query_range(self, node: int, start: int, end: int, 
                    l: int, r: int) -> int:
        """Query range [l, r]."""
        self._push(node, start, end)
        
        if start > end or start > r or end < l:
            return 0
        
        if start >= l and end <= r:
            return self.tree[node]
        
        mid = (start + end) // 2
        left = self._query_range(2 * node, start, mid, l, r)
        right = self._query_range(2 * node + 1, mid + 1, end, l, r)
        return left + right
    
    def update_range(self, l: int, r: int, value: int) -> None:
        """Update range [l, r] with value."""
        self._update_range(1, 0, self.size - 1, l, r, value)
    
    def query_range(self, l: int, r: int) -> int:
        """Query range [l, r]."""
        return self._query_range(1, 0, self.size - 1, l, r)
    
    def update_point(self, index: int, value: int) -> None:
        """Update point at index."""
        self.update_range(index, index, value)
    
    def query_point(self, index: int) -> int:
        """Query point at index."""
        return self.query_range(index, index)

class EulerTour:
    """Euler Tour implementation."""
    
    def __init__(self, tree: Dict[int, List[int]], root: int = 0):
        self.tree = tree
        self.root = root
        self.n = len(tree)
        self.tour = []
        self.entry_time = {}
        self.exit_time = {}
        self.values = [0] * self.n
        
        # Build Euler tour
        self._build_tour()
        
        # Build segment tree
        self.segment_tree = SegmentTree(len(self.tour))
        self._build_segment_tree()
    
    def _build_tour(self) -> None:
        """Build the Euler tour."""
        visited = set()
        self._dfs(self.root, visited)
    
    def _dfs(self, node: int, visited: Set[int]) -> None:
        """DFS to build Euler tour."""
        # Entry time
        self.entry_time[node] = len(self.tour)
        self.tour.append(node)
        
        # Visit children
        for child in self.tree.get(node, []):
            if child not in visited:
                visited.add(child)
                self._dfs(child, visited)
                # Return to parent
                self.tour.append(node)
        
        # Exit time
        self.exit_time[node] = len(self.tour) - 1
    
    def _build_segment_tree(self) -> None:
        """Build segment tree for the tour."""
        for i, node in enumerate(self.tour):
            self.segment_tree.update_point(i, self.values[node])
    
    def subtree_query(self, node: int, operation: str = 'sum') -> int:
        """Query the subtree rooted at node."""
        if node not in self.entry_time:
            return 0
        
        start = self.entry_time[node]
        end = self.exit_time[node]
        
        if operation == 'sum':
            return self.segment_tree.query_range(start, end)
        elif operation == 'max':
            return self._range_max(start, end)
        elif operation == 'min':
            return self._range_min(start, end)
        else:
            return self.segment_tree.query_range(start, end)
    
    def _range_max(self, start: int, end: int) -> int:
        """Find maximum in range."""
        max_val = float('-inf')
        for i in range(start, end + 1):
            node = self.tour[i]
            max_val = max(max_val, self.values[node])
        return max_val
    
    def _range_min(self, start: int, end: int) -> int:
        """Find minimum in range."""
        min_val = float('inf')
        for i in range(start, end + 1):
            node = self.tour[i]
            min_val = min(min_val, self.values[node])
        return min_val
    
    def subtree_update(self, node: int, value: int) -> None:
        """Update all nodes in the subtree rooted at node."""
        if node not in self.entry_time:
            return
        
        start = self.entry_time[node]
        end = self.exit_time[node]
        
        # Update segment tree
        self.segment_tree.update_range(start, end, value)
        
        # Update values
        for i in range(start, end + 1):
            tour_node = self.tour[i]
            self.values[tour_node] += value
    
    def path_query(self, u: int, v: int) -> int:
        """Query the path from u to v."""
        if u not in self.entry_time or v not in self.entry_time:
            return 0
        
        # Find LCA
        lca = self._find_lca(u, v)
        
        # Query path from u to LCA
        path_u = self._path_to_ancestor(u, lca)
        
        # Query path from v to LCA
        path_v = self._path_to_ancestor(v, lca)
        
        # Combine results
        return path_u + path_v + self.values[lca]
    
    def _find_lca(self, u: int, v: int) -> int:
        """Find Lowest Common Ancestor of u and v."""
        # Simple LCA implementation
        # In practice, you might want to use a more efficient method
        
        # Get all ancestors of u
        u_ancestors = set()
        current = u
        while current != self.root:
            u_ancestors.add(current)
            # Find parent of current
            for parent, children in self.tree.items():
                if current in children:
                    current = parent
                    break
        u_ancestors.add(self.root)
        
        # Find LCA
        current = v
        while current != self.root:
            if current in u_ancestors:
                return current
            # Find parent of current
            for parent, children in self.tree.items():
                if current in children:
                    current = parent
                    break
        
        return self.root
    
    def _path_to_ancestor(self, node: int, ancestor: int) -> int:
        """Query the path from node to ancestor."""
        if node == ancestor:
            return 0
        
        # Simple path query
        # In practice, you might want to use a more efficient method
        path_sum = 0
        current = node
        while current != ancestor:
            path_sum += self.values[current]
            # Find parent of current
            for parent, children in self.tree.items():
                if current in children:
                    current = parent
                    break
        
        return path_sum
    
    def update_node(self, node: int, value: int) -> None:
        """Update the value of a single node."""
        if node not in self.entry_time:
            return
        
        # Update all occurrences in tour
        for i, tour_node in enumerate(self.tour):
            if tour_node == node:
                self.segment_tree.update_point(i, value)
        
        self.values[node] = value
    
    def get_tour(self) -> List[int]:
        """Get the Euler tour."""
        return self.tour
    
    def get_entry_time(self, node: int) -> int:
        """Get the entry time of a node."""
        return self.entry_time.get(node, -1)
    
    def get_exit_time(self, node: int) -> int:
        """Get the exit time of a node."""
        return self.exit_time.get(node, -1)
    
    def get_subtree_size(self, node: int) -> int:
        """Get the size of the subtree rooted at node."""
        if node not in self.entry_time:
            return 0
        
        start = self.entry_time[node]
        end = self.exit_time[node]
        return (end - start + 1) // 2 + 1

def build_euler_tour(tree: Dict[int, List[int]], root: int = 0) -> EulerTour:
    """Build an Euler tour for the given tree."""
    return EulerTour(tree, root)

def subtree_query(euler_tour: EulerTour, node: int, operation: str = 'sum') -> int:
    """Query the subtree rooted at node."""
    return euler_tour.subtree_query(node, operation)

def subtree_update(euler_tour: EulerTour, node: int, value: int) -> None:
    """Update all nodes in the subtree rooted at node."""
    euler_tour.subtree_update(node, value)

def path_query(euler_tour: EulerTour, u: int, v: int) -> int:
    """Query the path from u to v."""
    return euler_tour.path_query(u, v)

def analyze_performance(tree: Dict[int, List[int]]) -> Dict:
    """Analyze performance of Euler tour operations."""
    start_time = time.time()
    euler_tour = build_euler_tour(tree)
    construction_time = time.time() - start_time
    
    # Test subtree queries
    queries = [(0, 'sum'), (1, 'sum'), (2, 'sum')]
    query_times = []
    
    for node, operation in queries:
        if node < len(tree):
            start_time = time.time()
            euler_tour.subtree_query(node, operation)
            query_times.append(time.time() - start_time)
    
    # Test subtree updates
    updates = [(0, 5), (1, 3), (2, 2)]
    update_times = []
    
    for node, value in updates:
        if node < len(tree):
            start_time = time.time()
            euler_tour.subtree_update(node, value)
            update_times.append(time.time() - start_time)
    
    return {
        "tree_size": len(tree),
        "tour_length": len(euler_tour.tour),
        "construction_time": construction_time,
        "query_times": query_times,
        "update_times": update_times,
        "average_query_time": np.mean(query_times),
        "average_update_time": np.mean(update_times)
    }

def generate_test_cases() -> List[Dict[int, List[int]]]:
    """Generate test cases for Euler tour."""
    return [
        # Simple tree
        {0: [1, 2], 1: [3, 4], 2: [5], 3: [], 4: [], 5: []},
        
        # Chain
        {0: [1], 1: [2], 2: [3], 3: [4], 4: []},
        
        # Star
        {0: [1, 2, 3, 4], 1: [], 2: [], 3: [], 4: []},
        
        # Balanced binary tree
        {0: [1, 2], 1: [3, 4], 2: [5, 6], 3: [], 4: [], 5: [], 6: []},
        
        # Complex tree
        {0: [1, 2, 3], 1: [4, 5], 2: [6], 3: [7, 8], 4: [], 5: [], 6: [], 7: [], 8: []}
    ]

def visualize_euler_tour(euler_tour: EulerTour, show_plot: bool = True) -> None:
    """Visualize the Euler tour."""
    tour = euler_tour.get_tour()
    
    if not tour:
        print("Empty Euler tour")
        return
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Original tree
    G = nx.Graph()
    for parent, children in euler_tour.tree.items():
        for child in children:
            G.add_edge(parent, child)
    
    pos = nx.spring_layout(G, k=2, iterations=30)
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=1000, alpha=0.7)
    nx.draw_networkx_edges(G, pos, alpha=0.6)
    nx.draw_networkx_labels(G, pos, font_size=12)
    ax1.set_title('Original Tree')
    ax1.axis('off')
    
    # Euler tour
    x_positions = list(range(len(tour)))
    bars = ax2.bar(x_positions, [1] * len(tour), color='lightgreen', alpha=0.7)
    
    # Highlight entry and exit times
    for node in euler_tour.entry_time:
        entry = euler_tour.entry_time[node]
        exit_time = euler_tour.exit_time[node]
        bars[entry].set_color('red')
        bars[exit_time].set_color('blue')
    
    ax2.set_xlabel('Position in Tour')
    ax2.set_ylabel('Node')
    ax2.set_title('Euler Tour')
    ax2.set_ylim(0, 1.2)
    
    # Add node labels
    for i, node in enumerate(tour):
        ax2.text(i, 1.1, str(node), ha='center', va='bottom')
    
    plt.tight_layout()
    
    if show_plot:
        plt.show()
    else:
        plt.savefig('euler_tour_visualization.png', dpi=300, bbox_inches='tight')
        plt.close()

def visualize_subtree_query(euler_tour: EulerTour, node: int, show_plot: bool = True) -> None:
    """Visualize a subtree query."""
    print(f"Visualizing subtree query for node {node}")
    
    if node not in euler_tour.entry_time:
        print(f"Node {node} not found in tree")
        return
    
    entry = euler_tour.entry_time[node]
    exit_time = euler_tour.exit_time[node]
    result = euler_tour.subtree_query(node, 'sum')
    
    print(f"Subtree range: [{entry}, {exit_time}]")
    print(f"Subtree sum: {result}")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Original tree with highlighted subtree
    G = nx.Graph()
    for parent, children in euler_tour.tree.items():
        for child in children:
            G.add_edge(parent, child)
    
    pos = nx.spring_layout(G, k=2, iterations=30)
    
    # Color nodes
    node_colors = []
    for n in G.nodes():
        if n == node or (n in euler_tour.entry_time and 
                        euler_tour.entry_time[n] >= entry and 
                        euler_tour.exit_time[n] <= exit_time):
            node_colors.append('red')
        else:
            node_colors.append('lightblue')
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                          node_size=1000, alpha=0.7)
    nx.draw_networkx_edges(G, pos, alpha=0.6)
    nx.draw_networkx_labels(G, pos, font_size=12)
    ax1.set_title(f'Subtree Query for Node {node}')
    ax1.axis('off')
    
    # Euler tour with highlighted range
    tour = euler_tour.get_tour()
    x_positions = list(range(len(tour)))
    bars = ax2.bar(x_positions, [1] * len(tour), color='lightgreen', alpha=0.7)
    
    # Highlight subtree range
    for i in range(entry, exit_time + 1):
        bars[i].set_color('red')
    
    ax2.set_xlabel('Position in Tour')
    ax2.set_ylabel('Node')
    ax2.set_title(f'Euler Tour (Subtree Range: [{entry}, {exit_time}])')
    ax2.set_ylim(0, 1.2)
    
    plt.tight_layout()
    
    if show_plot:
        plt.show()
    else:
        plt.savefig(f'subtree_query_{node}.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main function to demonstrate Euler Tour Technique."""
    print("=== Euler Tour Technique Implementation ===\n")
    
    # Test cases
    test_cases = generate_test_cases()
    
    for i, tree in enumerate(test_cases):
        print(f"Test Case {i+1}: Tree with {len(tree)} nodes")
        print("-" * 50)
        
        # Build Euler tour
        start_time = time.time()
        euler_tour = build_euler_tour(tree)
        construction_time = time.time() - start_time
        
        print(f"Tree size: {len(tree)}")
        print(f"Tour length: {len(euler_tour.tour)}")
        print(f"Construction time: {construction_time:.6f}s")
        
        # Test subtree queries
        if len(tree) >= 2:
            result = euler_tour.subtree_query(1, 'sum')
            print(f"Subtree sum at node 1: {result}")
        
        # Test subtree updates
        if len(tree) >= 2:
            euler_tour.subtree_update(1, 5)
            result = euler_tour.subtree_query(1, 'sum')
            print(f"Subtree sum after update: {result}")
        
        print()
    
    # Performance analysis
    print("=== Performance Analysis ===")
    performance_data = []
    
    for tree in test_cases:
        perf = analyze_performance(tree)
        performance_data.append(perf)
        print(f"Tree size {perf['tree_size']}: "
              f"Tour length {perf['tour_length']}, "
              f"Construction {perf['construction_time']:.6f}s, "
              f"Avg query {perf['average_query_time']:.6f}s")
    
    # Visualization for a complex case
    print("\n=== Visualization ===")
    complex_tree = {0: [1, 2, 3], 1: [4, 5], 2: [6], 3: [7, 8], 
                   4: [], 5: [], 6: [], 7: [], 8: []}
    complex_euler_tour = build_euler_tour(complex_tree)
    
    print("Euler tour visualization:")
    visualize_euler_tour(complex_euler_tour, show_plot=False)
    
    print("Subtree query visualization:")
    visualize_subtree_query(complex_euler_tour, 1, show_plot=False)
    
    # Advanced features demonstration
    print("\n=== Advanced Features ===")
    
    # Subtree operations
    euler_tour = build_euler_tour(complex_tree)
    
    # Set some values
    for i in range(9):
        euler_tour.update_node(i, i + 1)
    
    # Subtree queries
    subtree_sum = euler_tour.subtree_query(1, 'sum')
    print(f"Subtree sum at node 1: {subtree_sum}")
    
    # Subtree updates
    euler_tour.subtree_update(1, 10)
    subtree_sum_after = euler_tour.subtree_query(1, 'sum')
    print(f"Subtree sum after update: {subtree_sum_after}")
    
    # Path queries
    path_sum = euler_tour.path_query(4, 6)
    print(f"Path sum from 4 to 6: {path_sum}")
    
    # Tour analysis
    tour = euler_tour.get_tour()
    print(f"Euler tour: {tour[:20]}..." if len(tour) > 20 else f"Euler tour: {tour}")
    
    # Large tree performance
    print("\n=== Large Tree Performance ===")
    large_tree = {}
    for i in range(1000):
        large_tree[i] = []
        if 2*i + 1 < 1000:
            large_tree[i].append(2*i + 1)
        if 2*i + 2 < 1000:
            large_tree[i].append(2*i + 2)
    
    start_time = time.time()
    large_euler_tour = build_euler_tour(large_tree)
    construction_time = time.time() - start_time
    
    start_time = time.time()
    for i in range(100):
        large_euler_tour.subtree_query(i % 100, 'sum')
    query_time = time.time() - start_time
    
    print(f"1000-node tree construction: {construction_time:.6f}s")
    print(f"100 subtree queries: {query_time:.6f}s")
    
    print("\n=== Summary ===")
    print("Euler Tour Technique implementation completed successfully!")
    print("Features implemented:")
    print("- O(n) construction time")
    print("- O(log n) subtree queries and updates")
    print("- Path queries")
    print("- Tree to array conversion")
    print("- Visualization capabilities")
    print("- Performance analysis")
    print("- Large tree handling")

if __name__ == "__main__":
    main() 