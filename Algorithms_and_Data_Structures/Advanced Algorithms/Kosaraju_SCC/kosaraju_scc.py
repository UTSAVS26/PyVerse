"""
Kosaraju's Algorithm for Strongly Connected Components

This module implements Kosaraju's algorithm for finding strongly connected components
in a directed graph using two depth-first searches.

Author: Algorithm Implementation
Date: 2024
"""

import time
import numpy as np
from typing import Dict, List, Set, Tuple, Optional
import matplotlib.pyplot as plt
import networkx as nx


def kosaraju_scc(graph: Dict[int, List[int]]) -> List[List[int]]:
    """
    Find strongly connected components using Kosaraju's algorithm.
    
    Args:
        graph: Adjacency list representation of the directed graph
        
    Returns:
        List of strongly connected components, where each component is a list of vertices
    """
    visited = set()
    order = []
    
    def dfs1(u: int) -> None:
        """First DFS to get finishing order."""
        visited.add(u)
        for v in graph.get(u, []):
            if v not in visited:
                dfs1(v)
        order.append(u)
    
    def dfs2(u: int, component: List[int]) -> None:
        """Second DFS to find SCC."""
        visited.add(u)
        component.append(u)
        for v in transpose.get(u, []):
            if v not in visited:
                dfs2(v, component)
    
    # First pass: get finishing order
    for u in graph:
        if u not in visited:
            dfs1(u)
    
    # Create transpose graph
    transpose = {}
    for u in graph:
        for v in graph[u]:
            if v not in transpose:
                transpose[v] = []
            transpose[v].append(u)
    
    # Second pass: find SCCs
    visited.clear()
    sccs = []
    for u in reversed(order):
        if u not in visited:
            component = []
            dfs2(u, component)
            sccs.append(component)
    
    return sccs


def kosaraju_scc_optimized(graph: Dict[int, List[int]]) -> List[List[int]]:
    """
    Optimized version of Kosaraju's algorithm with better memory management.
    
    Args:
        graph: Adjacency list representation of the directed graph
        
    Returns:
        List of strongly connected components
    """
    visited = set()
    order = []
    
    def dfs1(u: int) -> None:
        """First DFS to get finishing order."""
        visited.add(u)
        for v in graph.get(u, []):
            if v not in visited:
                dfs1(v)
        order.append(u)
    
    def dfs2(u: int, component: List[int]) -> None:
        """Second DFS to find SCC."""
        visited.add(u)
        component.append(u)
        for v in graph.get(u, []):
            # Check if edge exists in transpose
            if v in transpose and u in transpose[v]:
                if v not in visited:
                    dfs2(v, component)
    
    # First pass: get finishing order
    for u in graph:
        if u not in visited:
            dfs1(u)
    
    # Create transpose graph more efficiently
    transpose = {}
    for u in graph:
        for v in graph[u]:
            if v not in transpose:
                transpose[v] = set()
            transpose[v].add(u)
    
    # Second pass: find SCCs
    visited.clear()
    sccs = []
    for u in reversed(order):
        if u not in visited:
            component = []
            dfs2(u, component)
            sccs.append(component)
    
    return sccs


def analyze_performance(graph: Dict[int, List[int]]) -> Dict:
    """
    Analyze performance of Kosaraju's algorithm.
    
    Args:
        graph: Input graph
        
    Returns:
        Dictionary with performance metrics
    """
    start_time = time.time()
    sccs = kosaraju_scc(graph)
    execution_time = time.time() - start_time
    
    num_vertices = len(graph)
    num_edges = sum(len(neighbors) for neighbors in graph.values())
    num_sccs = len(sccs)
    largest_scc_size = max(len(scc) for scc in sccs) if sccs else 0
    
    return {
        'num_vertices': num_vertices,
        'num_edges': num_edges,
        'num_sccs': num_sccs,
        'largest_scc_size': largest_scc_size,
        'execution_time': execution_time,
        'sccs': sccs
    }


def generate_test_cases() -> List[Dict[int, List[int]]]:
    """
    Generate test cases for Kosaraju's algorithm.
    
    Returns:
        List of test graphs
    """
    test_cases = [
        # Simple cycle
        {
            0: [1],
            1: [2],
            2: [0]
        },
        # Multiple SCCs
        {
            0: [1],
            1: [2],
            2: [0, 3],
            3: [4],
            4: [5],
            5: [3]
        },
        # Disconnected components
        {
            0: [1],
            1: [0],
            2: [3],
            3: [2]
        },
        # Complex graph
        {
            0: [1, 2],
            1: [2, 3],
            2: [3, 4],
            3: [4, 5],
            4: [5, 0],
            5: [0, 1]
        },
        # Single vertex SCCs
        {
            0: [1],
            1: [2],
            2: [3],
            3: []
        },
        # Large SCC with bridges
        {
            0: [1, 2],
            1: [2, 3],
            2: [3, 4],
            3: [4, 5],
            4: [5, 0],
            5: [0, 1],
            6: [7],
            7: [8],
            8: [6]
        }
    ]
    
    return test_cases


def visualize_sccs(graph: Dict[int, List[int]], sccs: List[List[int]], 
                   show_plot: bool = True) -> None:
    """
    Visualize the graph and highlight strongly connected components.
    
    Args:
        graph: Input graph
        sccs: List of strongly connected components
        show_plot: Whether to display the plot
    """
    try:
        G = nx.DiGraph()
        
        # Add edges
        for u in graph:
            for v in graph[u]:
                G.add_edge(u, v)
        
        # Create layout
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Create color map for SCCs
        color_map = {}
        for i, scc in enumerate(sccs):
            color = plt.cm.tab10(i % 10)
            for vertex in scc:
                color_map[vertex] = color
        
        # Draw graph
        plt.figure(figsize=(12, 8))
        
        # Draw nodes
        node_colors = [color_map.get(node, 'lightgray') for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                              node_size=700, alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, edge_color='gray', 
                              arrows=True, arrowsize=20, alpha=0.6)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
        
        # Add legend
        legend_elements = []
        for i, scc in enumerate(sccs):
            color = plt.cm.tab10(i % 10)
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                            markerfacecolor=color, markersize=10,
                                            label=f'SCC {i+1}: {scc}'))
        
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
        plt.title("Kosaraju's Strongly Connected Components", fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        if show_plot:
            plt.show()
        
    except ImportError:
        print("Visualization requires matplotlib and networkx. Install with:")
        print("pip install matplotlib networkx")


def kosaraju_scc_with_visualization(graph: Dict[int, List[int]], 
                                   show_plot: bool = True) -> List[List[int]]:
    """
    Find SCCs using Kosaraju's algorithm and optionally visualize the result.
    
    Args:
        graph: Input graph
        show_plot: Whether to show visualization
        
    Returns:
        List of strongly connected components
    """
    sccs = kosaraju_scc(graph)
    
    if show_plot:
        visualize_sccs(graph, sccs, show_plot=True)
    
    return sccs


def verify_sccs(graph: Dict[int, List[int]], sccs: List[List[int]]) -> bool:
    """
    Verify that the found SCCs are correct.
    
    Args:
        graph: Input graph
        sccs: List of strongly connected components
        
    Returns:
        True if SCCs are valid, False otherwise
    """
    # Check that all vertices are included
    all_vertices = set()
    for scc in sccs:
        all_vertices.update(scc)
    
    if all_vertices != set(graph.keys()):
        return False
    
    # Check that each SCC is strongly connected
    for scc in sccs:
        if len(scc) == 1:
            continue
        
        # Check if every vertex can reach every other vertex in the SCC
        for u in scc:
            for v in scc:
                if u != v:
                    # Simple reachability check
                    visited = set()
                    stack = [u]
                    while stack:
                        current = stack.pop()
                        if current == v:
                            break
                        if current not in visited:
                            visited.add(current)
                            for neighbor in graph.get(current, []):
                                if neighbor not in visited:
                                    stack.append(neighbor)
                    else:
                        return False  # v not reachable from u
    
    return True


def compare_with_tarjan(graph: Dict[int, List[int]]) -> Dict:
    """
    Compare Kosaraju's algorithm with Tarjan's algorithm.
    
    Args:
        graph: Input graph
        
    Returns:
        Dictionary with comparison results
    """
    # Kosaraju's algorithm
    start_time = time.time()
    kosaraju_sccs = kosaraju_scc(graph)
    kosaraju_time = time.time() - start_time
    
    # Tarjan's algorithm (simplified for comparison)
    def tarjan_scc_simple(graph):
        disc = {}
        low = {}
        stack = []
        on_stack = set()
        time_counter = 0
        sccs = []
        
        def dfs(u):
            nonlocal time_counter
            disc[u] = low[u] = time_counter
            time_counter += 1
            stack.append(u)
            on_stack.add(u)
            
            for v in graph.get(u, []):
                if v not in disc:
                    dfs(v)
                    low[u] = min(low[u], low[v])
                elif v in on_stack:
                    low[u] = min(low[u], disc[v])
            
            if disc[u] == low[u]:
                scc = []
                while True:
                    v = stack.pop()
                    on_stack.remove(v)
                    scc.append(v)
                    if v == u:
                        break
                sccs.append(scc)
        
        for u in graph:
            if u not in disc:
                dfs(u)
        
        return sccs
    
    start_time = time.time()
    tarjan_sccs = tarjan_scc_simple(graph)
    tarjan_time = time.time() - start_time
    
    return {
        'kosaraju_time': kosaraju_time,
        'tarjan_time': tarjan_time,
        'kosaraju_sccs': kosaraju_sccs,
        'tarjan_sccs': tarjan_sccs,
        'results_match': sorted(kosaraju_sccs) == sorted(tarjan_sccs)
    }


def main():
    """Main function to demonstrate Kosaraju's algorithm."""
    print("=" * 60)
    print("KOSARAJU'S STRONGLY CONNECTED COMPONENTS ALGORITHM")
    print("=" * 60)
    
    # Test cases
    test_cases = generate_test_cases()
    
    for i, graph in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Graph: {graph}")
        
        # Find SCCs
        sccs = kosaraju_scc(graph)
        print(f"Strongly Connected Components: {sccs}")
        
        # Verify result
        is_valid = verify_sccs(graph, sccs)
        print(f"Verification: {'Valid Valid' if is_valid else 'Invalid Invalid'}")
        
        # Performance analysis
        metrics = analyze_performance(graph)
        print(f"Performance:")
        print(f"  - Vertices: {metrics['num_vertices']}")
        print(f"  - Edges: {metrics['num_edges']}")
        print(f"  - SCCs: {metrics['num_sccs']}")
        print(f"  - Largest SCC: {metrics['largest_scc_size']} vertices")
        print(f"  - Execution time: {metrics['execution_time']:.6f}s")
    
    # Comparison with Tarjan's algorithm
    print("\n" + "=" * 60)
    print("COMPARISON WITH TARJAN'S ALGORITHM")
    print("=" * 60)
    
    complex_graph = {
        0: [1, 2],
        1: [2, 3],
        2: [3, 4],
        3: [4, 5],
        4: [5, 0],
        5: [0, 1],
        6: [7],
        7: [6],
        8: [9],
        9: [8, 10],
        10: [8]
    }
    
    comparison = compare_with_tarjan(complex_graph)
    print(f"Complex Graph: {complex_graph}")
    print(f"Kosaraju's SCCs: {comparison['kosaraju_sccs']}")
    print(f"Tarjan's SCCs: {comparison['tarjan_sccs']}")
    print(f"Results match: {'Valid Yes' if comparison['results_match'] else 'Invalid No'}")
    print(f"Kosaraju time: {comparison['kosaraju_time']:.6f}s")
    print(f"Tarjan time: {comparison['tarjan_time']:.6f}s")
    
    # Interactive example
    print("\n" + "=" * 60)
    print("INTERACTIVE EXAMPLE")
    print("=" * 60)
    
    print(f"Complex Graph: {complex_graph}")
    
    # Find SCCs
    sccs = kosaraju_scc(complex_graph)
    print(f"SCCs: {sccs}")
    
    # Visualize
    try:
        visualize_sccs(complex_graph, sccs, show_plot=True)
    except ImportError:
        print("Visualization skipped (matplotlib/networkx not available)")
    
    print("\nAlgorithm completed successfully!")


if __name__ == "__main__":
    main() 