"""
Articulation Points and Bridges Algorithm

This module implements algorithms to find articulation points (cut vertices) and 
bridges (cut edges) in an undirected graph using depth-first search.

Author: Algorithm Implementation
Date: 2024
"""

import time
import numpy as np
from typing import Dict, List, Set, Tuple, Optional
import matplotlib.pyplot as plt
import networkx as nx


def find_articulation_points(graph: Dict[int, List[int]]) -> List[int]:
    """
    Find articulation points (cut vertices) in an undirected graph.
    
    Args:
        graph: Adjacency list representation of the undirected graph
        
    Returns:
        List of articulation points
    """
    disc = {}  # Discovery times
    low = {}   # Low-link values
    parent = {}  # Parent vertices
    articulation_points = set()
    time_counter = 0
    
    def dfs(u: int) -> None:
        """Depth-first search to find articulation points."""
        nonlocal time_counter
        disc[u] = low[u] = time_counter
        time_counter += 1
        children = 0
        
        for v in graph.get(u, []):
            if v not in disc:
                parent[v] = u
                children += 1
                dfs(v)
                low[u] = min(low[u], low[v])
                
                # Check if u is articulation point
                if parent[u] is None and children > 1:
                    articulation_points.add(u)
                elif parent[u] is not None and low[v] >= disc[u]:
                    articulation_points.add(u)
            elif v != parent[u]:
                low[u] = min(low[u], disc[v])
    
    # Process all vertices
    for u in graph:
        if u not in disc:
            parent[u] = None
            dfs(u)
    
    return list(articulation_points)


def find_bridges(graph: Dict[int, List[int]]) -> List[Tuple[int, int]]:
    """
    Find bridges (cut edges) in an undirected graph.
    
    Args:
        graph: Adjacency list representation of the undirected graph
        
    Returns:
        List of bridges as tuples (u, v)
    """
    disc = {}  # Discovery times
    low = {}   # Low-link values
    parent = {}  # Parent vertices
    bridges = []
    time_counter = 0
    
    def dfs(u: int) -> None:
        """Depth-first search to find bridges."""
        nonlocal time_counter
        disc[u] = low[u] = time_counter
        time_counter += 1
        
        for v in graph.get(u, []):
            if v not in disc:
                parent[v] = u
                dfs(v)
                low[u] = min(low[u], low[v])
                
                # Check if (u, v) is a bridge
                if low[v] > disc[u]:
                    bridges.append((u, v))
            elif v != parent[u]:
                low[u] = min(low[u], disc[v])
    
    # Process all vertices
    for u in graph:
        if u not in disc:
            parent[u] = None
            dfs(u)
    
    return bridges


def find_articulation_points_and_bridges(graph: Dict[int, List[int]]) -> Tuple[List[int], List[Tuple[int, int]]]:
    """
    Find both articulation points and bridges in a single pass.
    
    Args:
        graph: Adjacency list representation of the undirected graph
        
    Returns:
        Tuple of (articulation_points, bridges)
    """
    disc = {}  # Discovery times
    low = {}   # Low-link values
    parent = {}  # Parent vertices
    articulation_points = set()
    bridges = []
    time_counter = 0
    
    def dfs(u: int) -> None:
        """Depth-first search to find both articulation points and bridges."""
        nonlocal time_counter
        disc[u] = low[u] = time_counter
        time_counter += 1
        children = 0
        
        for v in graph.get(u, []):
            if v not in disc:
                parent[v] = u
                children += 1
                dfs(v)
                low[u] = min(low[u], low[v])
                
                # Check if u is articulation point
                if parent[u] is None and children > 1:
                    articulation_points.add(u)
                elif parent[u] is not None and low[v] >= disc[u]:
                    articulation_points.add(u)
                
                # Check if (u, v) is a bridge
                if low[v] > disc[u]:
                    bridges.append((u, v))
            elif v != parent[u]:
                low[u] = min(low[u], disc[v])
    
    # Process all vertices
    for u in graph:
        if u not in disc:
            parent[u] = None
            dfs(u)
    
    return list(articulation_points), bridges


def analyze_performance(graph: Dict[int, List[int]]) -> Dict:
    """
    Analyze performance of articulation points and bridges algorithms.
    
    Args:
        graph: Input graph
        
    Returns:
        Dictionary with performance metrics
    """
    start_time = time.time()
    articulation_points, bridges = find_articulation_points_and_bridges(graph)
    execution_time = time.time() - start_time
    
    num_vertices = len(graph)
    num_edges = sum(len(neighbors) for neighbors in graph.values()) // 2  # Undirected graph
    num_articulation_points = len(articulation_points)
    num_bridges = len(bridges)
    
    return {
        'num_vertices': num_vertices,
        'num_edges': num_edges,
        'num_articulation_points': num_articulation_points,
        'num_bridges': num_bridges,
        'execution_time': execution_time,
        'articulation_points': articulation_points,
        'bridges': bridges
    }


def generate_test_cases() -> List[Dict[int, List[int]]]:
    """
    Generate test cases for articulation points and bridges algorithms.
    
    Returns:
        List of test graphs
    """
    test_cases = [
        # Simple graph with articulation point
        {
            0: [1, 2],
            1: [0, 2],
            2: [0, 1, 3],
            3: [2, 4],
            4: [3]
        },
        # Graph with multiple articulation points
        {
            0: [1, 2],
            1: [0, 2],
            2: [0, 1, 3],
            3: [2, 4, 5],
            4: [3, 5],
            5: [3, 4, 6],
            6: [5]
        },
        # Graph with bridges
        {
            0: [1],
            1: [0, 2],
            2: [1, 3],
            3: [2, 4],
            4: [3]
        },
        # Complex graph
        {
            0: [1, 2],
            1: [0, 2, 3],
            2: [0, 1, 3],
            3: [1, 2, 4],
            4: [3, 5, 6],
            5: [4, 6],
            6: [4, 5, 7],
            7: [6]
        },
        # Tree structure (all edges are bridges)
        {
            0: [1],
            1: [0, 2, 3],
            2: [1],
            3: [1, 4],
            4: [3]
        },
        # Biconnected component (no articulation points)
        {
            0: [1, 2],
            1: [0, 2],
            2: [0, 1]
        }
    ]
    
    return test_cases


def visualize_critical_elements(graph: Dict[int, List[int]], 
                              articulation_points: List[int], 
                              bridges: List[Tuple[int, int]], 
                              show_plot: bool = True) -> None:
    """
    Visualize the graph and highlight articulation points and bridges.
    
    Args:
        graph: Input graph
        articulation_points: List of articulation points
        bridges: List of bridges
        show_plot: Whether to display the plot
    """
    try:
        G = nx.Graph()
        
        # Add edges
        for u in graph:
            for v in graph[u]:
                if u < v:  # Avoid duplicate edges in undirected graph
                    G.add_edge(u, v)
        
        # Create layout
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Draw graph
        plt.figure(figsize=(12, 8))
        
        # Draw edges
        edge_colors = []
        for edge in G.edges():
            if edge in bridges or (edge[1], edge[0]) in bridges:
                edge_colors.append('red')  # Bridges in red
            else:
                edge_colors.append('gray')
        
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, 
                              width=2, alpha=0.7)
        
        # Draw nodes
        node_colors = []
        for node in G.nodes():
            if node in articulation_points:
                node_colors.append('red')  # Articulation points in red
            else:
                node_colors.append('lightblue')
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                              node_size=700, alpha=0.8)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                      markersize=10, label=f'Articulation Points: {articulation_points}'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
                      markersize=10, label='Regular Vertices'),
            plt.Line2D([0], [0], color='red', linewidth=2, 
                      label=f'Bridges: {bridges}'),
            plt.Line2D([0], [0], color='gray', linewidth=2, 
                      label='Regular Edges')
        ]
        
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
        plt.title("Articulation Points and Bridges", fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        if show_plot:
            plt.show()
        
    except ImportError:
        print("Visualization requires matplotlib and networkx. Install with:")
        print("pip install matplotlib networkx")


def verify_articulation_points(graph: Dict[int, List[int]], 
                             articulation_points: List[int]) -> bool:
    """
    Verify that the found articulation points are correct.
    
    Args:
        graph: Input graph
        articulation_points: List of articulation points
        
    Returns:
        True if articulation points are valid, False otherwise
    """
    # For each articulation point, check if removing it increases number of components
    for ap in articulation_points:
        # Create graph without the articulation point
        temp_graph = {}
        for u in graph:
            if u != ap:
                temp_graph[u] = [v for v in graph[u] if v != ap]
        
        # Count components in original graph
        original_components = count_components(graph)
        
        # Count components in graph without articulation point
        new_components = count_components(temp_graph)
        
        if new_components <= original_components:
            return False
    
    return True


def verify_bridges(graph: Dict[int, List[int]], bridges: List[Tuple[int, int]]) -> bool:
    """
    Verify that the found bridges are correct.
    
    Args:
        graph: Input graph
        bridges: List of bridges
        
    Returns:
        True if bridges are valid, False otherwise
    """
    # For each bridge, check if removing it increases number of components
    for bridge in bridges:
        u, v = bridge
        
        # Create graph without the bridge
        temp_graph = {}
        for vertex in graph:
            temp_graph[vertex] = [neighbor for neighbor in graph[vertex] 
                                if not (vertex == u and neighbor == v) and 
                                   not (vertex == v and neighbor == u)]
        
        # Count components in original graph
        original_components = count_components(graph)
        
        # Count components in graph without bridge
        new_components = count_components(temp_graph)
        
        if new_components <= original_components:
            return False
    
    return True


def count_components(graph: Dict[int, List[int]]) -> int:
    """
    Count the number of connected components in a graph.
    
    Args:
        graph: Input graph
        
    Returns:
        Number of connected components
    """
    visited = set()
    components = 0
    
    def dfs(u: int) -> None:
        """Depth-first search to mark connected component."""
        visited.add(u)
        for v in graph.get(u, []):
            if v not in visited:
                dfs(v)
    
    for u in graph:
        if u not in visited:
            dfs(u)
            components += 1
    
    return components


def find_biconnected_components(graph: Dict[int, List[int]]) -> List[List[int]]:
    """
    Find biconnected components of the graph.
    
    Args:
        graph: Input graph
        
    Returns:
        List of biconnected components
    """
    disc = {}
    low = {}
    parent = {}
    stack = []
    biconnected_components = []
    time_counter = 0
    
    def dfs(u: int) -> None:
        """Depth-first search to find biconnected components."""
        nonlocal time_counter
        disc[u] = low[u] = time_counter
        time_counter += 1
        
        for v in graph.get(u, []):
            if v not in disc:
                parent[v] = u
                stack.append((u, v))
                dfs(v)
                low[u] = min(low[u], low[v])
                
                # If u is articulation point, pop edges until (u, v)
                if low[v] >= disc[u]:
                    component = []
                    while stack:
                        edge = stack.pop()
                        component.append(edge)
                        if edge == (u, v):
                            break
                    biconnected_components.append(component)
            elif v != parent[u] and disc[v] < disc[u]:
                stack.append((u, v))
                low[u] = min(low[u], disc[v])
    
    # Process all vertices
    for u in graph:
        if u not in disc:
            parent[u] = None
            dfs(u)
    
    return biconnected_components


def main():
    """Main function to demonstrate articulation points and bridges algorithms."""
    print("=" * 60)
    print("ARTICULATION POINTS AND BRIDGES ALGORITHM")
    print("=" * 60)
    
    # Test cases
    test_cases = generate_test_cases()
    
    for i, graph in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Graph: {graph}")
        
        # Find articulation points and bridges
        articulation_points, bridges = find_articulation_points_and_bridges(graph)
        print(f"Articulation Points: {articulation_points}")
        print(f"Bridges: {bridges}")
        
        # Verify results
        ap_valid = verify_articulation_points(graph, articulation_points)
        bridges_valid = verify_bridges(graph, bridges)
        print(f"Verification: AP {'Valid' if ap_valid else 'Invalid'}, "
              f"Bridges {'Valid' if bridges_valid else 'Invalid'}")
        
        # Performance analysis
        metrics = analyze_performance(graph)
        print(f"Performance:")
        print(f"  - Vertices: {metrics['num_vertices']}")
        print(f"  - Edges: {metrics['num_edges']}")
        print(f"  - Articulation Points: {metrics['num_articulation_points']}")
        print(f"  - Bridges: {metrics['num_bridges']}")
        print(f"  - Execution time: {metrics['execution_time']:.6f}s")
    
    # Interactive example
    print("\n" + "=" * 60)
    print("INTERACTIVE EXAMPLE")
    print("=" * 60)
    
    # Create a complex graph
    complex_graph = {
        0: [1, 2],
        1: [0, 2, 3],
        2: [0, 1, 3],
        3: [1, 2, 4],
        4: [3, 5, 6],
        5: [4, 6],
        6: [4, 5, 7],
        7: [6]
    }
    
    print(f"Complex Graph: {complex_graph}")
    
    # Find articulation points and bridges
    articulation_points, bridges = find_articulation_points_and_bridges(complex_graph)
    print(f"Articulation Points: {articulation_points}")
    print(f"Bridges: {bridges}")
    
    # Find biconnected components
    biconnected_components = find_biconnected_components(complex_graph)
    print(f"Biconnected Components: {len(biconnected_components)}")
    
    # Visualize
    try:
        visualize_critical_elements(complex_graph, articulation_points, bridges, show_plot=True)
    except ImportError:
        print("Visualization skipped (matplotlib/networkx not available)")
    
    print("\nAlgorithm completed successfully!")


if __name__ == "__main__":
    main() 