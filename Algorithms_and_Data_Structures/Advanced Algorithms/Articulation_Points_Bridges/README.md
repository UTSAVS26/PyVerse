# Articulation Points and Bridges

Algorithms to find articulation points (cut vertices) and bridges (cut edges) in an undirected graph using depth-first search.

## Overview

Articulation points and bridges are critical elements in graph connectivity. An articulation point is a vertex whose removal increases the number of connected components, while a bridge is an edge whose removal increases the number of connected components.

## Algorithm

### Core Concept
The algorithm uses depth-first search (DFS) and maintains:
- **disc[]**: Discovery time of each vertex
- **low[]**: Lowest vertex reachable from current vertex
- **parent[]**: Parent vertex in DFS tree

### Algorithm Steps
1. **Initialize**: Set discovery time and low-link value for each vertex
2. **DFS Traversal**: Perform depth-first search from each unvisited vertex
3. **Low-link Update**: Update low-link values during backtracking
4. **Articulation Point Detection**: Check if current vertex is articulation point
5. **Bridge Detection**: Check if current edge is bridge

### Key Insights
- **Articulation Point**: A vertex `u` is an articulation point if:
  - `u` is root and has more than one child in DFS tree, OR
  - `u` is not root and has a child `v` with `low[v] >= disc[u]`
- **Bridge**: An edge `(u, v)` is a bridge if `low[v] > disc[u]`

## Time Complexity

- **Time**: O(V + E) where V is vertices, E is edges
- **Space**: O(V) for recursion stack and arrays

## Implementations

### 1. Standard Algorithm
- Classic implementation using recursion
- Finds both articulation points and bridges
- Handles all graph types

### 2. Optimized Version
- Memory-efficient implementation
- Better handling of large graphs
- Improved edge case handling

### 3. Visualization Support
- Optional matplotlib integration
- Plots graph and highlights critical elements
- Interactive visualization

## Usage

```python
from articulation_points_bridges import find_articulation_points, find_bridges

# Basic usage
graph = {
    0: [1, 2],
    1: [0, 2],
    2: [0, 1, 3],
    3: [2, 4],
    4: [3]
}

articulation_points = find_articulation_points(graph)
bridges = find_bridges(graph)

print(f"Articulation points: {articulation_points}")  # [2, 3]
print(f"Bridges: {bridges}")  # [(2, 3), (3, 4)]

# With visualization
from articulation_points_bridges import visualize_critical_elements
visualize_critical_elements(graph, articulation_points, bridges, show_plot=True)
```

## Mathematical Background

### Articulation Point
An articulation point (cut vertex) is a vertex whose removal increases the number of connected components in the graph.

### Bridge
A bridge (cut edge) is an edge whose removal increases the number of connected components in the graph.

### DFS Tree Properties
- **Tree Edge**: Edge to unvisited vertex
- **Back Edge**: Edge to ancestor in DFS tree
- **Forward Edge**: Edge to descendant in DFS tree
- **Cross Edge**: Edge to visited vertex (not ancestor/descendant)

### Low-link Value
The low-link value of a vertex `u` is the smallest discovery time of any vertex reachable from `u` through:
1. A direct edge from `u`
2. A path through descendants of `u` in DFS tree
3. A back edge from any descendant of `u`

## Applications

### 1. Network Design
- **Network Reliability**: Identify critical nodes and links
- **Fault Tolerance**: Design redundant connections
- **Traffic Analysis**: Understand network bottlenecks

### 2. Social Network Analysis
- **Community Detection**: Find bridge users between communities
- **Influence Analysis**: Identify key influencers
- **Network Resilience**: Analyze network robustness

### 3. Transportation Planning
- **Road Network Analysis**: Find critical intersections
- **Public Transport**: Identify essential routes
- **Emergency Planning**: Plan evacuation routes

### 4. Computer Networks
- **Network Topology**: Design robust network architecture
- **Routing Protocols**: Optimize routing paths
- **Security Analysis**: Identify vulnerable points

## Performance Analysis

The implementation includes performance analysis:

```python
from articulation_points_bridges import analyze_performance

metrics = analyze_performance(graph)
print(f"Number of vertices: {metrics['num_vertices']}")
print(f"Number of edges: {metrics['num_edges']}")
print(f"Number of articulation points: {metrics['num_articulation_points']}")
print(f"Number of bridges: {metrics['num_bridges']}")
print(f"Execution time: {metrics['execution_time']:.6f}s")
```

## Visualization

The algorithm includes comprehensive visualization:

```python
from articulation_points_bridges import visualize_critical_elements

# Visualize graph, articulation points, and bridges
visualize_critical_elements(graph, articulation_points, bridges, show_plot=True)
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
def find_articulation_points(graph):
    """Find articulation points using DFS."""
    disc = {}  # Discovery times
    low = {}   # Low-link values
    parent = {}  # Parent vertices
    articulation_points = set()
    time = 0
    
    def dfs(u):
        nonlocal time
        disc[u] = low[u] = time
        time += 1
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
```

## Edge Cases

### Self-Loops
- Handles vertices with edges to themselves
- Correctly identifies articulation points

### Disconnected Components
- Processes all connected components
- Handles isolated vertices

### Large Graphs
- Memory-efficient implementation
- Handles graphs with millions of vertices
- Optional iterative version for deep recursion

## Comparison with Other Algorithms

| Algorithm | Time Complexity | Space Complexity | Advantages |
|-----------|----------------|------------------|------------|
| DFS-based | O(V + E) | O(V) | Optimal, single pass |
| Naive | O(V × (V + E)) | O(V) | Simple but inefficient |
| BFS-based | O(V + E) | O(V) | Alternative approach |
| Union-Find | O(V + E) | O(V) | Different approach |

## Historical Context

The DFS-based algorithm for finding articulation points and bridges was developed in the 1970s. It's based on Tarjan's work on depth-first search and low-link values, which was later extended to handle undirected graphs.

## Applications in Computer Science

### 1. Graph Theory
- **Connectivity Analysis**: Understanding graph structure
- **Component Analysis**: Find critical elements
- **Graph Decomposition**: Break complex graphs

### 2. Network Analysis
- **Internet Topology**: Analyze web connectivity
- **Social Networks**: Identify key influencers
- **Transportation Networks**: Find critical routes

### 3. Software Engineering
- **Dependency Analysis**: Find critical dependencies
- **Module Clustering**: Identify essential modules
- **Build Systems**: Optimize compilation order

### 4. Data Science
- **Clustering**: Group similar data points
- **Recommendation Systems**: Find bridge items
- **Anomaly Detection**: Identify critical patterns

## Advanced Topics

### Biconnected Components
- Find maximal biconnected subgraphs
- Each biconnected component has no articulation points
- Useful for network reliability analysis

### Edge Connectivity
- Find minimum number of edges to disconnect graph
- Related to bridge detection
- Important for network design

### Vertex Connectivity
- Find minimum number of vertices to disconnect graph
- Related to articulation point detection
- Critical for network security

### Parallel Implementation
- Process multiple components in parallel
- Use shared memory for large graphs
- Implement distributed version for very large graphs 