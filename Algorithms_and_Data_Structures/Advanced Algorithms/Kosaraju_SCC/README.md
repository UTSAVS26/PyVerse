# Kosaraju's Algorithm

An algorithm to find strongly connected components (SCCs) in a directed graph using two depth-first searches.

## Overview

Kosaraju's algorithm is an efficient method for finding strongly connected components in a directed graph. It uses two passes of depth-first search (DFS) and is conceptually simpler than Tarjan's algorithm, though it requires two traversals of the graph.

## Algorithm

### Core Concept
The algorithm is based on the key insight that if we reverse all edges in a graph, the strongly connected components remain the same, but the order in which we process vertices becomes crucial.

### Algorithm Steps
1. **First DFS**: Perform DFS on the original graph and push vertices to stack in finishing order
2. **Transpose Graph**: Create the transpose of the original graph (reverse all edges)
3. **Second DFS**: Perform DFS on the transpose graph in the order of vertices from stack
4. **SCC Detection**: Each DFS tree in the second pass forms an SCC

### Key Insight
The finishing order from the first DFS ensures that when we process vertices in the transpose graph, we start from "sink" components and work our way up to "source" components.

## Time Complexity

- **Time**: O(V + E) where V is vertices, E is edges
- **Space**: O(V) for recursion stack and arrays

## Implementations

### 1. Standard Kosaraju's Algorithm
- Two-pass DFS implementation
- Clear and intuitive approach
- Handles all graph types

### 2. Optimized Version
- Memory-efficient implementation
- Better handling of large graphs
- Improved stack management

### 3. Visualization Support
- Optional matplotlib integration
- Plots graph and highlights SCCs
- Shows both passes of the algorithm

## Usage

```python
from kosaraju_scc import kosaraju_scc

# Basic usage
graph = {
    0: [1],
    1: [2],
    2: [0, 3],
    3: [4],
    4: [5],
    5: [3]
}

sccs = kosaraju_scc(graph)
print(sccs)  # [[0, 1, 2], [3, 4, 5]]

# With visualization
from kosaraju_scc import kosaraju_scc_with_visualization
sccs = kosaraju_scc_with_visualization(graph, show_plot=True)
```

## Mathematical Background

### Strongly Connected Component
A strongly connected component (SCC) of a directed graph is a maximal subgraph where every vertex is reachable from every other vertex within the component.

### Graph Transpose
The transpose of a graph G is a graph G^T where:
- V(G^T) = V(G)
- E(G^T) = {(v, u) | (u, v) ∈ E(G)}

### DFS Finishing Order
The finishing order of vertices in DFS is crucial:
- Vertices in "sink" SCCs finish first
- Vertices in "source" SCCs finish last
- This order ensures proper SCC discovery

## Applications

### 1. Compiler Design
- **Control Flow Analysis**: Identify loops and basic blocks
- **Dead Code Elimination**: Remove unreachable code
- **Optimization**: Analyze program structure

### 2. Social Network Analysis
- **Community Detection**: Find tightly connected groups
- **Influence Propagation**: Analyze information flow
- **Network Clustering**: Identify cohesive subgroups

### 3. Web Crawling
- **Link Analysis**: Find related web pages
- **Site Structure**: Understand website organization
- **SEO Optimization**: Analyze internal linking

### 4. Circuit Design
- **Feedback Loops**: Identify cyclic dependencies
- **Timing Analysis**: Analyze signal propagation
- **Design Verification**: Check circuit correctness

## Performance Analysis

The implementation includes performance analysis:

```python
from kosaraju_scc import analyze_performance

metrics = analyze_performance(graph)
print(f"Number of vertices: {metrics['num_vertices']}")
print(f"Number of edges: {metrics['num_edges']}")
print(f"Number of SCCs: {metrics['num_sccs']}")
print(f"Execution time: {metrics['execution_time']:.6f}s")
print(f"Largest SCC size: {metrics['largest_scc_size']}")
```

## Visualization

The algorithm includes comprehensive visualization:

```python
from kosaraju_scc import kosaraju_scc_with_visualization

# Visualize graph and SCCs
sccs = kosaraju_scc_with_visualization(graph, show_plot=True)
```

## Requirements

- Python 3.7+
- NumPy (for numerical operations)
- Matplotlib (for visualization, optional)
- NetworkX (for graph operations, optional)

## Installation

```bash
pip install numpy matplotlib networkx
```

## Algorithm Details

### Standard Implementation
```python
def kosaraju_scc(graph):
    """Find strongly connected components using Kosaraju's algorithm."""
    visited = set()
    order = []
    
    def dfs1(u):
        """First DFS to get finishing order."""
        visited.add(u)
        for v in graph.get(u, []):
            if v not in visited:
                dfs1(v)
        order.append(u)
    
    def dfs2(u, component):
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
```

## Edge Cases

### Self-Loops
- Handles vertices with edges to themselves
- Correctly identifies single-vertex SCCs

### Disconnected Components
- Processes all connected components
- Handles isolated vertices

### Large Graphs
- Memory-efficient implementation
- Handles graphs with millions of vertices
- Efficient transpose graph creation

## Comparison with Other Algorithms

| Algorithm | Time Complexity | Space Complexity | Advantages |
|-----------|----------------|------------------|------------|
| Kosaraju's | O(V + E) | O(V) | Simple, intuitive |
| Tarjan's | O(V + E) | O(V) | Single DFS pass |
| Gabow's | O(V + E) | O(V) | Alternative approach |
| Naive | O(V³) | O(V²) | Simple but inefficient |

## Historical Context

Kosaraju's algorithm was developed by S. Rao Kosaraju in 1978. It was one of the first linear-time algorithms for finding strongly connected components and remains popular due to its conceptual simplicity and ease of implementation.

## Applications in Computer Science

### 1. Graph Theory
- **Component Analysis**: Understanding graph structure
- **Connectivity Testing**: Check if graph is strongly connected
- **Graph Decomposition**: Break complex graphs into simpler parts

### 2. Network Analysis
- **Internet Topology**: Analyze web connectivity
- **Social Networks**: Identify communities and influencers
- **Transportation Networks**: Analyze route connectivity

### 3. Software Engineering
- **Dependency Analysis**: Find circular dependencies
- **Module Clustering**: Group related components
- **Build Systems**: Optimize compilation order

### 4. Data Science
- **Clustering**: Group similar data points
- **Recommendation Systems**: Find related items
- **Anomaly Detection**: Identify unusual patterns

## Advantages and Disadvantages

### Advantages
- **Conceptual Simplicity**: Easy to understand and implement
- **Intuitive**: Two-pass approach is straightforward
- **Robust**: Handles all edge cases well
- **Parallelizable**: Second pass can be parallelized

### Disadvantages
- **Two Passes**: Requires two DFS traversals
- **Memory Overhead**: Needs to store transpose graph
- **Cache Performance**: May have poor cache locality

## Optimization Techniques

### Memory Optimization
- Use adjacency lists instead of matrices
- Implement lazy transpose graph creation
- Reuse memory for visited arrays

### Performance Optimization
- Use iterative DFS for large graphs
- Implement early termination for single SCC
- Optimize transpose graph construction

### Parallel Implementation
- Parallelize second DFS pass
- Use shared memory for large graphs
- Implement distributed version for very large graphs 