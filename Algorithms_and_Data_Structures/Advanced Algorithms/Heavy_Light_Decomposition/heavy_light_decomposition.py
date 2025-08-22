import time
import numpy as np
from typing import Dict, List, Set, Tuple, Optional
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

class HLD:
    """Heavy Light Decomposition implementation."""
    
    def __init__(self, tree: Dict[int, List[int]], root: int = 0):
        self.tree = tree
        self.root = root
        self.n = len(tree)
        
        # DFS arrays
        self.parent = [-1] * self.n
        self.depth = [0] * self.n
        self.size = [1] * self.n
        self.heavy = [-1] * self.n
        
        # HLD arrays
        self.chain_head = [0] * self.n
        self.chain_id = [0] * self.n
        self.pos_in_chain = [0] * self.n
        self.chain_size = [0] * self.n
        
        # Values
        self.values = [0] * self.n
        
        # Build HLD
        self._dfs_size(self.root)
        self._dfs_decompose(self.root, self.root)
        self._build_segment_trees()
    
    def _dfs_size(self, u: int) -> int:
        """DFS to compute subtree sizes and find heavy children."""
        max_child_size = 0
        heavy_child = -1
        
        for v in self.tree[u]:
            if v != self.parent[u]:
                self.parent[v] = u
                self.depth[v] = self.depth[u] + 1
                child_size = self._dfs_size(v)
                self.size[u] += child_size
                
                if child_size > max_child_size:
                    max_child_size = child_size
                    heavy_child = v
        
        self.heavy[u] = heavy_child
        return self.size[u]
    
    def _dfs_decompose(self, u: int, head: int) -> None:
        """DFS to decompose tree into heavy paths."""
        self.chain_head[u] = head
        
        if not hasattr(self, 'chains'):
            self.chains = []
        
        if u == head:
            self.chains.append([])
            self.chain_id[u] = len(self.chains) - 1
        else:
            self.chain_id[u] = self.chain_id[head]
        
        self.pos_in_chain[u] = self.chain_size[head]
        self.chain_size[head] += 1
        
        self.chains[self.chain_id[u]].append(u)
        
        # Process heavy child first
        if self.heavy[u] != -1:
            self._dfs_decompose(self.heavy[u], head)
        
        # Process light children
        for v in self.tree[u]:
            if v != self.parent[u] and v != self.heavy[u]:
                self._dfs_decompose(v, v)
    
    def _build_segment_trees(self) -> None:
        """Build segment trees for each heavy path."""
        self.segment_trees = []
        for chain in self.chains:
            if chain:
                st = SegmentTree(len(chain))
                self.segment_trees.append(st)
    
    def _get_lca(self, u: int, v: int) -> int:
        """Find Lowest Common Ancestor of u and v."""
        while self.chain_head[u] != self.chain_head[v]:
            if self.depth[self.chain_head[u]] < self.depth[self.chain_head[v]]:
                u, v = v, u
            u = self.parent[self.chain_head[u]]
        
        return u if self.depth[u] < self.depth[v] else v
    
    def _query_chain(self, u: int, v: int) -> int:
        """Query the chain segment [u..v], where u is ancestor of v (same chain)."""
        chain_id = self.chain_id[u]
        st = self.segment_trees[chain_id]
        left = self.pos_in_chain[u]
        right = self.pos_in_chain[v]
        return st.query_range(left, right)

    def _update_chain(self, u: int, v: int, value: int) -> None:
        """Add `value` over the chain segment [u..v], where u is ancestor of v (same chain)."""
        chain_id = self.chain_id[u]
        st = self.segment_trees[chain_id]
        left = self.pos_in_chain[u]
        right = self.pos_in_chain[v]
        st.update_range(left, right, value)
    def path_query(self, u: int, v: int, operation: str = 'sum') -> int:
        """Query the path from u to v."""
        lca = self._get_lca(u, v)
        
        # Query from u to LCA
        result_u = 0
        while u != lca:
            if self.chain_head[u] == self.chain_head[lca]:
                result_u += self._query_chain(lca, u)
                break
            else:
                result_u += self._query_chain(self.chain_head[u], u)
                u = self.parent[self.chain_head[u]]
        
        # Query from v to LCA
        result_v = 0
        while v != lca:
            if self.chain_head[v] == self.chain_head[lca]:
                result_v += self._query_chain(lca, v)
                break
            else:
                result_v += self._query_chain(self.chain_head[v], v)
                v = self.parent[self.chain_head[v]]
        
        # Combine results
        if operation == 'sum':
            return result_u + result_v + self.values[lca]
        elif operation == 'max':
            return max(result_u, result_v, self.values[lca])
        elif operation == 'min':
            return min(result_u, result_v, self.values[lca])
        else:
            return result_u + result_v + self.values[lca]
    
    def path_update(self, u: int, v: int, value: int) -> None:
        """Update all nodes on the path from u to v."""
        lca = self._get_lca(u, v)
        
        # Update from u to LCA
        while u != lca:
            if self.chain_head[u] == self.chain_head[lca]:
                self._update_chain(lca, u, value)
                break
            else:
                self._update_chain(self.chain_head[u], u, value)
                u = self.parent[self.chain_head[u]]
        
        # Update from v to LCA
        while v != lca:
            if self.chain_head[v] == self.chain_head[lca]:
                self._update_chain(lca, v, value)
                break
            else:
                self._update_chain(self.chain_head[v], v, value)
                v = self.parent[self.chain_head[v]]
        
        # Update LCA
        self.values[lca] += value
    
    def subtree_query(self, node: int, operation: str = 'sum') -> int:
        """Query the subtree rooted at node."""
        # For simplicity, we'll use a different approach
        # In practice, you might want to maintain subtree sizes
        result = self.values[node]
        
        def dfs_subtree(u: int) -> int:
            total = self.values[u]
            for v in self.tree[u]:
                if v != self.parent[u]:
                    total += dfs_subtree(v)
            return total
        
        return dfs_subtree(node)
    
    def point_update(self, node: int, value: int) -> None:
        """Update a single node."""
        self.values[node] = value
    
    def get_heavy_paths(self) -> List[List[int]]:
        """Get all heavy paths."""
        return self.chains

def build_hld(tree: Dict[int, List[int]], root: int = 0) -> HLD:
    """Build Heavy Light Decomposition for the given tree."""
    return HLD(tree, root)

def analyze_performance(tree: Dict[int, List[int]]) -> Dict:
    """Analyze performance of HLD operations."""
    start_time = time.time()
    hld = build_hld(tree)
    construction_time = time.time() - start_time
    
    # Test path queries
    queries = [(0, 1), (1, 2), (0, 2)]
    query_times = []
    
    for u, v in queries:
        if u < len(tree) and v < len(tree):
            start_time = time.time()
            hld.path_query(u, v)
            query_times.append(time.time() - start_time)
    
    # Test path updates
    updates = [(0, 1, 5), (1, 2, 3)]
    update_times = []
    
    for u, v, val in updates:
        if u < len(tree) and v < len(tree):
            start_time = time.time()
            hld.path_update(u, v, val)
            update_times.append(time.time() - start_time)
    
    return {
        "tree_size": len(tree),
        "heavy_paths": len(hld.chains),
        "construction_time": construction_time,
        "query_times": query_times,
        "update_times": update_times,
        "max_chain_length": max(len(chain) for chain in hld.chains) if hld.chains else 0
    }

def generate_test_cases() -> List[Dict[int, List[int]]]:
    """Generate test cases for HLD."""
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

def visualize_hld(hld: HLD, show_plot: bool = True) -> None:
    """Visualize the Heavy Light Decomposition."""
    G = nx.Graph()
    
    # Add nodes
    for i in range(hld.n):
        G.add_node(i)
    
    # Add edges
    for u in hld.tree:
        for v in hld.tree[u]:
            G.add_edge(u, v)
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Color nodes by chain
    node_colors = [hld.chain_id[i] for i in range(hld.n)]
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                          node_size=1000, alpha=0.7, cmap=plt.cm.Set3)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.6)
    
    # Draw labels
    labels = {i: f"{i}\nChain:{hld.chain_id[i]}" for i in range(hld.n)}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    
    plt.title("Heavy Light Decomposition")
    plt.axis('off')
    
    if show_plot:
        plt.show()
    else:
        plt.savefig('heavy_light_decomposition.png', dpi=300, bbox_inches='tight')
        plt.close()

def visualize_path_query(hld: HLD, u: int, v: int, show_plot: bool = True) -> None:
    """Visualize a path query."""
    G = nx.Graph()
    
    # Add nodes
    for i in range(hld.n):
        G.add_node(i)
    
    # Add edges
    for u_node in hld.tree:
        for v_node in hld.tree[u_node]:
            G.add_edge(u_node, v_node)
    
    # Find path
    lca = hld._get_lca(u, v)
    path_nodes = set()
    
    # Add nodes from u to LCA
    current = u
    while current != lca:
        path_nodes.add(current)
        current = hld.parent[current]
    path_nodes.add(lca)
    
    # Add nodes from v to LCA
    current = v
    while current != lca:
        path_nodes.add(current)
        current = hld.parent[current]
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(G, k=2, iterations=30)
    
    # Color nodes
    node_colors = ['red' if i in path_nodes else 'lightblue' for i in range(hld.n)]
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                          node_size=1000, alpha=0.8)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    plt.title(f"Path Query: {u} to {v}")
    plt.axis('off')
    
    if show_plot:
        plt.show()
    else:
        plt.savefig(f'path_query_{u}_{v}.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main function to demonstrate Heavy Light Decomposition."""
    print("=== Heavy Light Decomposition Implementation ===\n")
    
    # Test cases
    test_cases = generate_test_cases()
    
    for i, tree in enumerate(test_cases):
        print(f"Test Case {i+1}: Tree with {len(tree)} nodes")
        print("-" * 50)
        
        # Build HLD
        start_time = time.time()
        hld = build_hld(tree)
        construction_time = time.time() - start_time
        
        print(f"Tree size: {len(tree)}")
        print(f"Heavy paths: {len(hld.chains)}")
        print(f"Construction time: {construction_time:.6f}s")
        
        # Test path queries
        if len(tree) >= 2:
            result = hld.path_query(0, 1)
            print(f"Path query (0->1): {result}")
        
        # Test path updates
        if len(tree) >= 2:
            hld.path_update(0, 1, 5)
            result = hld.path_query(0, 1)
            print(f"After update (0->1): {result}")
        
        print()
    
    # Performance analysis
    print("=== Performance Analysis ===")
    performance_data = []
    
    for tree in test_cases:
        perf = analyze_performance(tree)
        performance_data.append(perf)
        print(f"Tree size {perf['tree_size']}: "
              f"{perf['heavy_paths']} heavy paths, "
              f"{perf['construction_time']:.6f}s construction")
    
    # Visualization for a complex case
    print("\n=== Visualization ===")
    complex_tree = {0: [1, 2, 3], 1: [4, 5], 2: [6], 3: [7, 8], 
                   4: [], 5: [], 6: [], 7: [], 8: []}
    hld = build_hld(complex_tree)
    
    print("HLD structure visualization:")
    visualize_hld(hld, show_plot=False)
    
    print("Path query visualization:")
    visualize_path_query(hld, 4, 7, show_plot=False)
    
    # Advanced features demonstration
    print("\n=== Advanced Features ===")
    
    # Heavy paths
    heavy_paths = hld.get_heavy_paths()
    print(f"Heavy paths: {heavy_paths}")
    
    # Subtree query
    subtree_sum = hld.subtree_query(1)
    print(f"Subtree sum at node 1: {subtree_sum}")
    
    # Point update
    hld.point_update(2, 10)
    point_value = hld.values[2]
    print(f"Point update at node 2: {point_value}")
    
    print("\n=== Summary ===")
    print("Heavy Light Decomposition implementation completed successfully!")
    print("Features implemented:")
    print("- Linear time construction")
    print("- Path queries and updates")
    print("- Subtree queries")
    print("- Point updates")
    print("- Visualization capabilities")
    print("- Performance analysis")

if __name__ == "__main__":
    main() 