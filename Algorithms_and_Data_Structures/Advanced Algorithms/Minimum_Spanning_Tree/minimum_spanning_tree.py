"""
Minimum Spanning Tree Algorithms

This module implements both Kruskal's and Prim's algorithms for finding
minimum spanning trees in weighted undirected graphs.

Author: Algorithm Implementation
Date: 2024
"""

import time
import heapq
# Replace top-level imports
- import numpy as np
- from typing import Dict, List, Set, Tuple, Optional
- import matplotlib.pyplot as plt
 from typing import Dict, List, Tuple

# Inside the visualization function
 def visualize_mst(...):
-    try:
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
        G = nx.Graph()
     except ImportError:
         print("Visualization dependencies not installed; skipping visualization.")
     # …rest of visualization logic…


class UnionFind:
    """Union-Find data structure for Kruskal's algorithm."""
    
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x: int) -> int:
        """Find the root of the set containing x."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x: int, y: int) -> bool:
        """Union the sets containing x and y."""
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True


def kruskal_mst(graph: Dict[int, List[Tuple[int, int]]]) -> List[Tuple[int, int, int]]:
    """
    Find minimum spanning tree using Kruskal's algorithm.
    
    Args:
        graph: Adjacency list with weights as (vertex, weight) tuples
        
    Returns:
        List of MST edges as (u, v, weight) tuples
    """
    edges = []
    for u in graph:
        for v, weight in graph[u]:
            if u < v:  # Avoid duplicate edges in undirected graph
                edges.append((weight, u, v))
    
    edges.sort()  # Sort by weight
    mst = []
    uf = UnionFind(len(graph))
    
    for weight, u, v in edges:
        if uf.union(u, v):
            mst.append((u, v, weight))
            if len(mst) == len(graph) - 1:
                break
    
    return mst


def prim_mst(graph: Dict[int, List[Tuple[int, int]]]) -> List[Tuple[int, int, int]]:
    """
    Find minimum spanning tree using Prim's algorithm.
    
    Args:
        graph: Adjacency list with weights as (vertex, weight) tuples
        
    Returns:
        List of MST edges as (u, v, weight) tuples
    """
    mst: List[Tuple[int, int, int]] = []
    visited = set()
    # Pick an arbitrary start vertex
    start = next(iter(graph)) if graph else None
    if start is None:
        return mst
    pq = [(0, start, None)]  # (weight, vertex, parent)
    
    while pq and len(visited) < len(graph):
        weight, u, parent = heapq.heappop(pq)
        
        if u in visited:
            continue
        
        visited.add(u)
        if parent is not None:
            # normalize edge orientation so comparisons aren’t order-dependent
            a, b = (parent, u) if parent <= u else (u, parent)
            mst.append((a, b, weight))
        
        for v, w in graph[u]:
            if v not in visited:
                heapq.heappush(pq, (w, v, u))
    
    return mst


def analyze_performance(graph: Dict[int, List[Tuple[int, int]]]) -> Dict:
    """
    Analyze performance of both MST algorithms.
    
    Args:
        graph: Input graph
        
    Returns:
        Dictionary with performance metrics
    """
    # Kruskal's algorithm
    start_time = time.time()
    kruskal_mst_edges = kruskal_mst(graph)
    kruskal_time = time.time() - start_time
    
    # Prim's algorithm
    start_time = time.time()
    prim_mst_edges = prim_mst(graph)
    prim_time = time.time() - start_time
    
    # Calculate MST weight
    kruskal_weight = sum(weight for _, _, weight in kruskal_mst_edges)
    prim_weight = sum(weight for _, _, weight in prim_mst_edges)
    
    num_vertices = len(graph)
    num_edges = sum(len(neighbors) for neighbors in graph.values()) // 2
    
    return {
        'num_vertices': num_vertices,
        'num_edges': num_edges,
        'kruskal_time': kruskal_time,
        'prim_time': prim_time,
        'kruskal_weight': kruskal_weight,
        'prim_weight': prim_weight,
        'kruskal_mst': kruskal_mst_edges,
        'prim_mst': prim_mst_edges
    }


def generate_test_cases() -> List[Dict[int, List[Tuple[int, int]]]]:
    """
    Generate test cases for MST algorithms.
    
    Returns:
        List of test graphs
    """
    test_cases = [
        # Simple graph
        {
            0: [(1, 4), (2, 3)],
            1: [(0, 4), (2, 1), (3, 2)],
            2: [(0, 3), (1, 1), (3, 4)],
            3: [(1, 2), (2, 4)]
        },
        # Larger graph
        {
            0: [(1, 2), (2, 4), (3, 6)],
            1: [(0, 2), (2, 1), (3, 3)],
            2: [(0, 4), (1, 1), (3, 5)],
            3: [(0, 6), (1, 3), (2, 5)]
        },
        # Graph with multiple components
        {
            0: [(1, 2), (2, 3)],
            1: [(0, 2), (2, 1)],
            2: [(0, 3), (1, 1)],
            3: [(4, 1), (5, 2)],
            4: [(3, 1), (5, 3)],
            5: [(3, 2), (4, 3)]
        },
        # Complex graph
        {
            0: [(1, 4), (2, 8), (3, 5)],
            1: [(0, 4), (2, 11), (3, 9)],
            2: [(0, 8), (1, 11), (3, 7)],
            3: [(0, 5), (1, 9), (2, 7), (4, 6)],
            4: [(3, 6), (5, 2)],
            5: [(4, 2), (6, 1)],
            6: [(5, 1), (7, 3)],
            7: [(6, 3)]
        }
    ]
    
    return test_cases


def visualize_mst(graph: Dict[int, List[Tuple[int, int]]], 
                 mst_edges: List[Tuple[int, int, int]], 
                 algorithm: str = "Kruskal", 
                 show_plot: bool = True) -> None:
    """
    Visualize the graph and highlight the minimum spanning tree.
    
    Args:
        graph: Input graph
        mst_edges: List of MST edges
        algorithm: Name of the algorithm used
        show_plot: Whether to display the plot
    """
    try:
        G = nx.Graph()
        
        # Add all edges
        for u in graph:
            for v, weight in graph[u]:
                if u < v:  # Avoid duplicate edges
                    G.add_edge(u, v, weight=weight)
        
        # Create layout
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Draw graph
        plt.figure(figsize=(12, 8))
        
        # Draw all edges in light gray
        nx.draw_networkx_edges(G, pos, edge_color='lightgray', 
                              width=1, alpha=0.3)
        
        # Draw MST edges in red
        mst_graph = nx.Graph()
        for u, v, weight in mst_edges:
            mst_graph.add_edge(u, v, weight=weight)
        
        nx.draw_networkx_edges(mst_graph, pos, edge_color='red', 
                              width=3, alpha=0.8)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                              node_size=700, alpha=0.8)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
        
        # Add edge labels for MST
        edge_labels = {(u, v): weight for u, v, weight in mst_edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, 
                                    font_size=10, font_color='red')
        
        # Calculate MST weight
        mst_weight = sum(weight for _, _, weight in mst_edges)
        
        plt.title(f"{algorithm}'s MST (Weight: {mst_weight})", 
                 fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        if show_plot:
            plt.show()
        
    except ImportError:
        print("Visualization requires matplotlib and networkx. Install with:")
        print("pip install matplotlib networkx")


def verify_mst(graph: Dict[int, List[Tuple[int, int]]], 
              mst_edges: List[Tuple[int, int, int]]) -> bool:
    """
    Verify that the MST is correct.
    
    Args:
        graph: Input graph
        mst_edges: List of MST edges
        
    Returns:
        True if MST is valid, False otherwise
    """
    # Check that MST has V-1 edges
    if len(mst_edges) != len(graph) - 1:
        return False
    
    # Check that MST is connected
    mst_graph = {}
    for u, v, _ in mst_edges:
        if u not in mst_graph:
            mst_graph[u] = []
        if v not in mst_graph:
            mst_graph[v] = []
        mst_graph[u].append(v)
        mst_graph[v].append(u)
    
    # Check connectivity using DFS
    visited = set()
    
    def dfs(u):
        visited.add(u)
        for v in mst_graph.get(u, []):
            if v not in visited:
                dfs(v)
    
    if mst_graph:
        start_vertex = next(iter(mst_graph))
        dfs(start_vertex)
        if len(visited) != len(graph):
            return False
    
    return True


def compare_mst_algorithms(graph: Dict[int, List[Tuple[int, int]]]) -> Dict:
    """
    Compare Kruskal's and Prim's algorithms.
    
    Args:
        graph: Input graph
        
    Returns:
        Dictionary with comparison results
    """
    metrics = analyze_performance(graph)
    
    return {
        'kruskal_time': metrics['kruskal_time'],
        'prim_time': metrics['prim_time'],
        'kruskal_weight': metrics['kruskal_weight'],
        'prim_weight': metrics['prim_weight'],
        'kruskal_mst': metrics['kruskal_mst'],
        'prim_mst': metrics['prim_mst'],
        'weights_match': metrics['kruskal_weight'] == metrics['prim_weight'],
        'edges_match': sorted((min(u, v), max(u, v), w) for u, v, w in metrics['kruskal_mst']) ==
                       sorted((min(u, v), max(u, v), w) for u, v, w in metrics['prim_mst'])
    }


def find_all_msts(graph: Dict[int, List[Tuple[int, int]]]) -> List[List[Tuple[int, int, int]]]:
    """
    Find all minimum spanning trees (if multiple exist).
    
    Args:
        graph: Input graph
        
    Returns:
        List of all MSTs
    """
    # This is a simplified implementation
    # In practice, finding all MSTs is more complex
    
    # Get one MST first
    mst = kruskal_mst(graph)
    mst_weight = sum(weight for _, _, weight in mst)
    
    # Find edges with same weight as MST edges
    all_edges = []
    for u in graph:
        for v, weight in graph[u]:
            if u < v:
                all_edges.append((weight, u, v))
    
    # Group edges by weight
    weight_groups = {}
    for weight, u, v in all_edges:
        if weight not in weight_groups:
            weight_groups[weight] = []
        weight_groups[weight].append((u, v))
    
    # For now, return the single MST
    # A full implementation would find all combinations
    return [mst]


def main():
    """Main function to demonstrate MST algorithms."""
    print("=" * 60)
    print("MINIMUM SPANNING TREE ALGORITHMS")
    print("=" * 60)
    
    # Test cases
    test_cases = generate_test_cases()
    
    for i, graph in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Graph: {graph}")
        
        # Compare algorithms
        comparison = compare_mst_algorithms(graph)
        
        print(f"Kruskal's MST: {comparison['kruskal_mst']}")
        print(f"Prim's MST: {comparison['prim_mst']}")
        print(f"Weights match: {'Valid Yes' if comparison['weights_match'] else 'Invalid No'}")
        print(f"Edges match: {'Valid Yes' if comparison['edges_match'] else 'Invalid No'}")
        print(f"Kruskal weight: {comparison['kruskal_weight']}")
        print(f"Prim weight: {comparison['prim_weight']}")
        print(f"Kruskal time: {comparison['kruskal_time']:.6f}s")
        print(f"Prim time: {comparison['prim_time']:.6f}s")
        
        # Verify MST
        kruskal_valid = verify_mst(graph, comparison['kruskal_mst'])
        prim_valid = verify_mst(graph, comparison['prim_mst'])
        print(f"Verification: Kruskal {'Valid Valid' if kruskal_valid else 'Invalid Invalid'}, "
              f"Prim {'Valid Valid' if prim_valid else 'Invalid Invalid'}")
    
    # Interactive example
    print("\n" + "=" * 60)
    print("INTERACTIVE EXAMPLE")
    print("=" * 60)
    
    # Create a complex graph
    complex_graph = {
        0: [(1, 4), (2, 8), (3, 5)],
        1: [(0, 4), (2, 11), (3, 9)],
        2: [(0, 8), (1, 11), (3, 7)],
        3: [(0, 5), (1, 9), (2, 7), (4, 6)],
        4: [(3, 6), (5, 2)],
        5: [(4, 2), (6, 1)],
        6: [(5, 1), (7, 3)],
        7: [(6, 3)]
    }
    
    print(f"Complex Graph: {complex_graph}")
    
    # Find MSTs
    kruskal_mst_edges = kruskal_mst(complex_graph)
    prim_mst_edges = prim_mst(complex_graph)
    
    print(f"Kruskal's MST: {kruskal_mst_edges}")
    print(f"Prim's MST: {prim_mst_edges}")
    
    # Visualize
    try:
        visualize_mst(complex_graph, kruskal_mst_edges, "Kruskal", show_plot=True)
        visualize_mst(complex_graph, prim_mst_edges, "Prim", show_plot=True)
    except ImportError:
        print("Visualization skipped (matplotlib/networkx not available)")
    
    print("\nAlgorithm completed successfully!")


if __name__ == "__main__":
    main() 