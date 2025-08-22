import time
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict

class SegmentTree:
    """Segment Tree for efficient range queries"""
    
    def __init__(self, size: int):
        self.size = size
        self.tree = [0] * (4 * size)
        self.lazy = [0] * (4 * size)
    
    def _build(self, arr: List[int], node: int, start: int, end: int) -> None:
        """Build segment tree"""
        if start == end:
            self.tree[node] = arr[start]
            return
        
        mid = (start + end) // 2
        self._build(arr, 2 * node, start, mid)
        self._build(arr, 2 * node + 1, mid + 1, end)
        self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]
    
    def build(self, arr: List[int]) -> None:
        """Build segment tree from array"""
        self._build(arr, 1, 0, self.size - 1)
    
    def _query(self, node: int, start: int, end: int, left: int, right: int) -> int:
        """Query range sum"""
        if right < start or left > end:
            return 0
        
        if left <= start and right >= end:
            return self.tree[node]
        
        mid = (start + end) // 2
        return (self._query(2 * node, start, mid, left, right) +
                self._query(2 * node + 1, mid + 1, end, left, right))
    
    def query(self, left: int, right: int) -> int:
        """Query range sum"""
        return self._query(1, 0, self.size - 1, left, right)
    
    def _update(self, node: int, start: int, end: int, index: int, value: int) -> None:
        """Update single element"""
        if start == end:
            self.tree[node] = value
            return
        
        mid = (start + end) // 2
        if index <= mid:
            self._update(2 * node, start, mid, index, value)
        else:
            self._update(2 * node + 1, mid + 1, end, index, value)
        
        self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]
    
    def update(self, index: int, value: int) -> None:
        """Update element at index"""
        self._update(1, 0, self.size - 1, index, value)

class HeavyPathDecomposition:
    """Enhanced Heavy Path Decomposition for tree queries"""
    
    def __init__(self, tree: List[List[int]], values: Optional[List[int]] = None):
        self.tree = tree
        self.n = len(tree)
        self.values = values if values else [0] * self.n
        self.heavy = [-1] * self.n
        self.parent = [-1] * self.n
        self.depth = [0] * self.n
        self.size = [0] * self.n
        self.chain_head = [-1] * self.n
        self.pos = [-1] * self.n
        self.operation_log = []
        self._decompose()
        self._build_segment_trees()
    
    def _dfs_size(self, u: int, p: int) -> int:
        """DFS to compute subtree sizes"""
        self.parent[u] = p
        self.size[u] = 1
        
        for v in self.tree[u]:
            if v != p:
                self.depth[v] = self.depth[u] + 1
                self.size[u] += self._dfs_size(v, u)
        
        return self.size[u]
    
    def _dfs_decompose(self, u: int, head: int) -> None:
        """DFS to decompose tree into heavy paths"""
        self.chain_head[u] = head
    # In __init__ (or wherever operation_log is initialized):
        self.position_counter = 0
        self.operation_log = []

    def _dfs_decompose(self, u: int, head: int) -> None:
        """DFS to decompose tree into heavy paths"""
        self.chain_head[u] = head
-       self.pos[u] = len(self.operation_log)
       self.pos[u] = self.position_counter
       self.position_counter += 1
       self.operation_log.append(u)
        
        # Find heavy child
        heavy_child = -1
        max_size = 0
        
        for v in self.tree[u]:
            if v != self.parent[u] and self.size[v] > max_size:
                max_size = self.size[v]
                heavy_child = v
        
        if heavy_child != -1:
            self.heavy[u] = heavy_child
            self._dfs_decompose(heavy_child, head)
        
        # Process light children
        for v in self.tree[u]:
            if v != self.parent[u] and v != heavy_child:
                self._dfs_decompose(v, v)
    
    def _decompose(self) -> None:
        """Build heavy path decomposition"""
        self._dfs_size(0, -1)
        self._dfs_decompose(0, 0)
    
    def _build_segment_trees(self) -> None:
        """Build segment trees for each heavy path"""
        self.segment_trees = {}
        
        # Group nodes by chain head
        chains = defaultdict(list)
        for i in range(self.n):
            chains[self.chain_head[i]].append(i)
        
        # Build segment tree for each chain
        for head, nodes in chains.items():
            if nodes:
                # Sort nodes by position
                nodes.sort(key=lambda x: self.pos[x])
                chain_values = [self.values[node] for node in nodes]
                
                st = SegmentTree(len(chain_values))
                st.build(chain_values)
                self.segment_trees[head] = st
    
    def path_query(self, u: int, v: int, operation: str = 'sum') -> int:
        """Query operation on path from u to v"""
        if u == v:
            return self.values[u]
        
        # Get LCA
        lca = self._get_lca(u, v)
        
        # Query path from u to LCA
        result_u = self._path_query_to_lca(u, lca, operation)
        
        # Query path from v to LCA
        result_v = self._path_query_to_lca(v, lca, operation)
        
        # Combine results based on operation
        if operation == 'sum':
        # Combine results based on operation
        if operation == 'sum':
            # result_u and result_v each include values[lca], so subtract one copy
            return result_u + result_v - self.values[lca]
        elif operation == 'min':
            return min(result_u, result_v, self.values[lca])
        elif operation == 'max':
            return max(result_u, result_v, self.values[lca])
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    def _path_query_to_lca(self, u: int, lca: int, operation: str) -> int:
        """Query path from u to LCA"""
        result = 0 if operation == 'sum' else (float('inf') if operation == 'min' else float('-inf'))
        
        while u != lca:
            if self.chain_head[u] == self.chain_head[lca]:
                # Same heavy path
                if operation == 'sum':
                    result += self._query_chain(u, lca)
                elif operation == 'min':
                    result = min(result, self._query_chain_min(u, lca))
                elif operation == 'max':
                    result = max(result, self._query_chain_max(u, lca))
                break
            else:
                # Different heavy paths
                head = self.chain_head[u]
                if operation == 'sum':
                    result += self._query_chain(u, head)
                elif operation == 'min':
                    result = min(result, self._query_chain_min(u, head))
                elif operation == 'max':
                    result = max(result, self._query_chain_max(u, head))
                u = self.parent[head]
        
        return result
    
    def _query_chain(self, u: int, v: int) -> int:
        """Query sum in chain from u to v"""
        head = self.chain_head[u]
        if head not in self.segment_trees:
            return 0
        
     def _query_chain(self, u: int, v: int) -> int:
         """Query sum in chain from u to v"""
         head = self.chain_head[u]
         if head not in self.segment_trees:
             return 0
         
-        # Get positions in chain
-        pos_u = self.pos[u]
        # Get positions relative to chain
        chain_nodes = [node for node in range(self.n) if self.chain_head[node] == head]
        chain_nodes.sort(key=lambda x: self.pos[x])
        pos_u = chain_nodes.index(u)
        pos_v = chain_nodes.index(v)
         
         # Query segment tree
         return self.segment_trees[head].query(min(pos_u, pos_v), max(pos_u, pos_v))
    
    def _query_chain_min(self, u: int, v: int) -> int:
        """Query minimum in chain from u to v"""
        # Simplified implementation
        current = u
        min_val = self.values[u]
        
        while current != v and current != -1:
            min_val = min(min_val, self.values[current])
            current = self.parent[current]
        
        return min_val
    
    def _query_chain_max(self, u: int, v: int) -> int:
        """Query maximum in chain from u to v"""
        # Simplified implementation
        current = u
        max_val = self.values[u]
        
        while current != v and current != -1:
            max_val = max(max_val, self.values[current])
            current = self.parent[current]
        
        return max_val
    
    def subtree_query(self, u: int, operation: str = 'sum') -> int:
        """Query operation on subtree rooted at u"""
        # Get all nodes in subtree using DFS
        subtree_nodes = self._get_subtree_nodes(u)
        
        if operation == 'sum':
            return sum(self.values[node] for node in subtree_nodes)
        elif operation == 'min':
            return min(self.values[node] for node in subtree_nodes)
        elif operation == 'max':
            return max(self.values[node] for node in subtree_nodes)
        else:
            return sum(self.values[node] for node in subtree_nodes)
    
    def _get_subtree_nodes(self, u: int) -> List[int]:
        """Get all nodes in subtree rooted at u"""
        nodes = [u]
        stack = [u]
        
        while stack:
            current = stack.pop()
            for v in self.tree[current]:
                if v != self.parent[current]:
                    nodes.append(v)
                    stack.append(v)
        
        return nodes
    
    def update_node(self, u: int, value: int) -> None:
        """Update value at node u"""
        self.values[u] = value
        
        # Update segment tree if node is in a chain
        head = self.chain_head[u]
        if head in self.segment_trees:
         # Update segment tree if node is in a chain
         head = self.chain_head[u]
         if head in self.segment_trees:
             chain_nodes = [node for node in range(self.n) if self.chain_head[node] == head]
             chain_nodes.sort(key=lambda x: self.pos[x])
             pos = chain_nodes.index(u)
             self.segment_trees[head].update(pos, value)
    
    def update_path(self, u: int, v: int, value: int) -> None:
        """Update values on path from u to v"""
        if u == v:
            self.update_node(u, value)
            return
        
        # Get LCA
        lca = self._get_lca(u, v)
        
        # Update path from u to LCA
        self._update_path_to_lca(u, lca, value)
        
        # Update path from v to LCA
        self._update_path_to_lca(v, lca, value)
        
        # Update LCA
        self.update_node(lca, value)
    
    def _update_path_to_lca(self, u: int, lca: int, value: int) -> None:
        """Update path from u to LCA"""
        while u != lca:
            if self.chain_head[u] == self.chain_head[lca]:
                # Same heavy path
                self._update_chain(u, lca, value)
                break
            else:
                # Different heavy paths
                head = self.chain_head[u]
                self._update_chain(u, head, value)
                u = self.parent[head]
    
    def _update_chain(self, u: int, v: int, value: int) -> None:
        """Update values in chain from u to v"""
        current = u
        while current != v and current != -1:
            self.update_node(current, value)
            current = self.parent[current]
    
    def get_lca(self, u: int, v: int) -> int:
        """Get lowest common ancestor of u and v"""
        return self._get_lca(u, v)
    
    def _get_lca(self, u: int, v: int) -> int:
        """Get LCA using binary lifting (simplified)"""
        # Simplified LCA implementation
        # In practice, you would use binary lifting for O(log n) queries
        
        # Make sure u and v are at same depth
        while self.depth[u] > self.depth[v]:
            u = self.parent[u]
        while self.depth[v] > self.depth[u]:
            v = self.parent[v]
        
        # Find LCA
        while u != v:
            u = self.parent[u]
            v = self.parent[v]
        
        return u
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the decomposition"""
        return {
            'tree_size': self.n,
            'heavy_paths': len(set(self.chain_head)),
            'segment_trees': len(self.segment_trees),
            'total_operations': len(self.operation_log),
            'max_depth': max(self.depth),
            'values': self.values
        }
    
    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze performance metrics"""
        if not self.operation_log:
            return {}
        
++ b/Algorithms_and_Data_Structures/Advanced Algorithms/Heavy_Path_Decomposition_Enhanced/heavy_path_decomposition.py
@@ class HeavyPathDecomposition:
     def __init__(self, ...):
        # Track actual query/update operations instead of node IDs
        self.query_count = 0
        self.update_count = 0
        self.operation_log = []
@@
    def path_query(self, u: int, v: int, operation: str = 'sum') -> int:
        """Query operation on path from u to v"""
        self.query_count += 1
        # ... rest of implementation ...

    def update_node(self, u: int, value: int) -> None:
        """Update value at node u"""
        self.update_count += 1
        # ... rest of implementation ...
@@
     def analyze_performance(self) -> Dict[str, Any]:
         """Analyze performance metrics"""
-        if not self.operation_log:
-            return {}
-        
-        operations = [op for op in self.operation_log]
-        timestamps = list(range(len(operations)))
-        
-        # Calculate operation frequencies
-        op_counts = defaultdict(int)
-        for op in operations:
-            op_counts[type(op).__name__] += 1
-        
-        return {
-            'total_operations': len(self.operation_log),
-            'operation_counts': dict(op_counts),
-            'average_interval': np.mean(timestamps) if timestamps else 0,
-            'total_time': timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0,
-            'decomposition_statistics': self.get_statistics()
        # Return counts of the two actual operations
        return {
            'total_operations': self.query_count + self.update_count,
            'operation_counts': {
                'queries': self.query_count,
                'updates': self.update_count
            },
            'decomposition_statistics': self.get_statistics()
        }
        
        return {
            'total_operations': len(self.operation_log),
            'operation_counts': dict(op_counts),
            'average_interval': np.mean(timestamps) if timestamps else 0,
            'total_time': timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0,
            'decomposition_statistics': self.get_statistics()
        }

def build_heavy_path_decomposition(tree: List[List[int]], values: Optional[List[int]] = None) -> HeavyPathDecomposition:
    """Build heavy path decomposition from tree"""
    return HeavyPathDecomposition(tree, values)

def generate_test_cases() -> List[Dict[str, Any]]:
    """Generate test cases for Heavy Path Decomposition"""
    test_cases = []
    
    # Test case 1: Basic operations
    tree1 = [[1, 2], [0, 3, 4], [0], [1, 5], [1], [3]]
    values1 = [1, 2, 3, 4, 5, 6]
    hpd1 = HeavyPathDecomposition(tree1, values1)
    
    test_cases.append({
        'name': 'Basic Operations',
        'decomposition': hpd1,
        'tree': tree1,
        'values': values1,
        'expected_path_queries': {
            (0, 5): 13,  # Sum of path 0->1->3->5 = 1+2+4+6
            (2, 4): 10   # Sum of path 2->0->1->4
        }
    })
    
    # Test case 2: Subtree queries
    tree2 = [[1, 2], [0, 3, 4], [0], [1, 5], [1], [3]]
    values2 = [1, 2, 3, 4, 5, 6]
    hpd2 = HeavyPathDecomposition(tree2, values2)
    
    test_cases.append({
        'name': 'Subtree Queries',
        'decomposition': hpd2,
        'tree': tree2,
        'values': values2,
        'expected_subtree_queries': {
            1: 17,  # Sum of subtree at node 1
            2: 3    # Sum of subtree at node 2
        }
    })
    
    # Test case 3: Edge cases
    tree3 = [[]]
    values3 = [1]
    hpd3 = HeavyPathDecomposition(tree3, values3)
    
    test_cases.append({
        'name': 'Single Node',
        'decomposition': hpd3,
        'tree': tree3,
        'values': values3,
        'expected_path_queries': {
            (0, 0): 1
        }
    })
    
    return test_cases

def visualize_heavy_path_decomposition(hpd: HeavyPathDecomposition, show_plot: bool = True) -> None:
    """Visualize the heavy path decomposition"""
    if not hpd.tree:
        print("Empty tree")
        return
    
    # Create networkx graph
    G = nx.Graph()
    pos = {}
    
    def add_nodes(node: int, x: float = 0, y: float = 0, level: int = 0, width: float = 2.0):
        if node == -1:
            return
        
        G.add_node(node, value=hpd.values[node], depth=hpd.depth[node], 
                  size=hpd.size[node], chain_head=hpd.chain_head[node])
        pos[node] = (x, -y)
        
        # Add edges to children
        for child in hpd.tree[node]:
            if child != hpd.parent[node]:
                G.add_edge(node, child)
                # Position child
                child_x = x + (list(hpd.tree[node]).index(child) - len(hpd.tree[node])/2) * width
                add_nodes(child, child_x, y + 1, level + 1, width/2)
    
    add_nodes(0, 0, 0)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Tree structure with heavy paths
    nx.draw(G, pos, ax=ax1, with_labels=True, 
           node_color='lightblue', node_size=1000,
           arrows=False)
    
    # Add node labels
    labels = {node: f"{node}\n{hpd.values[node]}" for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, ax=ax1)
    
    ax1.set_title('Tree Structure with Node Values')
    
    # Plot 2: Heavy path statistics
    stats = hpd.get_statistics()
    metrics = ['Tree Size', 'Heavy Paths', 'Segment Trees', 'Max Depth']
    values = [stats['tree_size'], stats['heavy_paths'], 
              stats['segment_trees'], stats['max_depth']]
    
    ax2.bar(metrics, values, color=['skyblue', 'lightgreen', 'orange', 'red'])
    ax2.set_title('Decomposition Statistics')
    ax2.set_ylabel('Value')
    
    plt.tight_layout()
    if show_plot:
        plt.show()

def visualize_performance_metrics(hpd: HeavyPathDecomposition, show_plot: bool = True) -> None:
    """Visualize performance metrics"""
    if not hpd.operation_log:
        print("No operations to analyze")
        return
    
    # Extract data
    operations = [type(op).__name__ for op in hpd.operation_log]
    timestamps = list(range(len(hpd.operation_log)))
    
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
    
    # Plot 3: Tree size vs operations
    tree_sizes = [hpd.n] * len(timestamps)
    operation_counts = list(range(len(timestamps)))
    
    ax3.plot(range(len(tree_sizes)), tree_sizes, 'g-', marker='s', label='Tree Size')
    ax3.plot(range(len(operation_counts)), operation_counts, 'r-', marker='o', label='Operations')
    ax3.set_title('Tree Size vs Operations')
    ax3.set_xlabel('Operation Index')
    ax3.set_ylabel('Count')
    ax3.legend()
    
    # Plot 4: Decomposition statistics
    stats = hpd.get_statistics()
    metrics = ['Tree Size', 'Heavy Paths', 'Segment Trees', 'Max Depth']
    values = [stats['tree_size'], stats['heavy_paths'], 
              stats['segment_trees'], stats['max_depth']]
    
    ax4.bar(metrics, values, color=['orange', 'red', 'purple', 'green'])
    ax4.set_title('Decomposition Statistics')
    ax4.set_ylabel('Value')
    
    plt.tight_layout()
    if show_plot:
        plt.show()

def main():
    """Main function to demonstrate Heavy Path Decomposition"""
    print("=== Enhanced Heavy Path Decomposition Demo ===\n")
    
    # Create tree and decomposition
    tree = [[1, 2], [0, 3, 4], [0], [1, 5], [1], [3]]
    values = [1, 2, 3, 4, 5, 6]
    hpd = HeavyPathDecomposition(tree, values)
    
    print("1. Basic Operations:")
    print(f"   Tree size: {hpd.n}")
    print(f"   Values: {hpd.values}")
    print(f"   Max depth: {max(hpd.depth)}")
    
    print("\n2. Path Queries:")
    path_queries = [(0, 5), (2, 4), (1, 3)]
    for u, v in path_queries:
        sum_result = hpd.path_query(u, v, 'sum')
        min_result = hpd.path_query(u, v, 'min')
        max_result = hpd.path_query(u, v, 'max')
        print(f"   Path[{u}, {v}] - Sum: {sum_result}, Min: {min_result}, Max: {max_result}")
    
    print("\n3. Subtree Queries:")
    subtree_nodes = [1, 2, 3]
    for u in subtree_nodes:
        sum_result = hpd.subtree_query(u, 'sum')
        min_result = hpd.subtree_query(u, 'min')
        max_result = hpd.subtree_query(u, 'max')
        print(f"   Subtree[{u}] - Sum: {sum_result}, Min: {min_result}, Max: {max_result}")
    
    print("\n4. LCA Queries:")
    lca_pairs = [(2, 4), (1, 5), (0, 3)]
    for u, v in lca_pairs:
        lca = hpd.get_lca(u, v)
        print(f"   LCA({u}, {v}) = {lca}")
    
    print("\n5. Update Operations:")
    print("   Updating node 3 to value 10")
    hpd.update_node(3, 10)
    print(f"   New value at node 3: {hpd.values[3]}")
    
    print("\n6. Performance Analysis:")
    perf = hpd.analyze_performance()
    print(f"   Tree size: {perf.get('decomposition_statistics', {}).get('tree_size', 0)}")
    print(f"   Total operations: {perf.get('total_operations', 0)}")
    print(f"   Operation counts: {perf.get('operation_counts', {})}")
    
    print("\n7. Decomposition Statistics:")
    stats = hpd.get_statistics()
    print(f"   Tree size: {stats['tree_size']}")
    print(f"   Heavy paths: {stats['heavy_paths']}")
    print(f"   Segment trees: {stats['segment_trees']}")
    print(f"   Max depth: {stats['max_depth']}")
    
    print("\n8. Visualization:")
    print("   Generating visualizations...")
    
    # Generate visualizations
    visualize_heavy_path_decomposition(hpd, show_plot=False)
    visualize_performance_metrics(hpd, show_plot=False)
    
    print("   Visualizations created successfully!")
    
    print("\n9. Test Cases:")
    test_cases = generate_test_cases()
    print(f"   Generated {len(test_cases)} test cases")
    
    for i, test_case in enumerate(test_cases):
        print(f"   Test case {i+1}: {test_case['name']}")
    
    print("\n=== Demo Complete ===")
    
    # Show one visualization
    visualize_heavy_path_decomposition(hpd)

if __name__ == "__main__":
    main() 