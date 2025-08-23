# Minimum Spanning Tree

Algorithms to find the minimum spanning tree (MST) of a weighted undirected graph. Implements both Kruskal's and Prim's algorithms with visualization.

## Overview

A minimum spanning tree (MST) is a subset of the edges of a connected, undirected graph that connects all the vertices together, without any cycles and with the minimum possible total edge weight.

## Algorithms

### Kruskal's Algorithm
1. **Sort Edges**: Sort all edges by weight in ascending order
2. **Initialize**: Start with empty MST and disjoint-set data structure
3. **Process Edges**: For each edge in sorted order:
   - If adding the edge doesn't create a cycle, add it to MST
   - Use union-find to detect cycles efficiently
4. **Complete**: Stop when MST has V-1 edges (where V is number of vertices)

### Prim's Algorithm
1. **Initialize**: Start with any vertex and empty MST
2. **Grow Tree**: Repeatedly add the minimum weight edge that connects a vertex in MST to a vertex outside MST
3. **Priority Queue**: Use min-heap to efficiently find minimum weight edges
4. **Complete**: Stop when all vertices are included in MST

## Time Complexity

### Kruskal's Algorithm
- **Time**: O(E log E) where E is number of edges
- **Space**: O(V) for union-find data structure

### Prim's Algorithm
- **Time**: O(E log V) with binary heap, O(V²) with array
- **Space**: O(V) for priority queue and visited array

## Implementations

### 1. Kruskal's Algorithm
- Uses union-find data structure
- Efficient for sparse graphs
- Easy to implement and understand

### 2. Prim's Algorithm
- Uses priority queue (min-heap)
- Efficient for dense graphs
- Better for graphs with many edges

### 3. Visualization Support
- Optional matplotlib integration
- Plots graph and highlights MST
- Shows algorithm progress step-by-step

## Usage

```python
from minimum_spanning_tree import kruskal_mst, prim_mst

# Basic usage
graph = {
    0: [(1, 4), (2, 3)],
    1: [(0, 4), (2, 1), (3, 2)],
    2: [(0, 3), (1, 1), (3, 4)],
    3: [(1, 2), (2, 4)]
}

# Using Kruskal's algorithm
mst_kruskal = kruskal_mst(graph)
print(f"Kruskal MST: {mst_kruskal}")

# Using Prim's algorithm
mst_prim = prim_mst(graph)
print(f"Prim MST: {mst_prim}")

# With visualization
from minimum_spanning_tree import visualize_mst
visualize_mst(graph, mst_kruskal, algorithm="Kruskal", show_plot=True)
```

## Mathematical Background

### Minimum Spanning Tree
A minimum spanning tree of a weighted graph is a spanning tree whose sum of edge weights is as small as possible.

### Properties of MST
- **Uniqueness**: If all edge weights are distinct, MST is unique
- **Cut Property**: For any cut in the graph, the minimum weight edge crossing the cut is in MST
- **Cycle Property**: For any cycle in the graph, the maximum weight edge is not in MST

### Cut Property
If we partition the vertices into two sets, the minimum weight edge crossing the cut must be in the MST.

### Cycle Property
If we add an edge to MST, it creates a cycle. The maximum weight edge in this cycle is not in MST.

## Applications

### 1. Network Design
- **Computer Networks**: Design efficient network topology
- **Telecommunications**: Plan cable/fiber optic networks
- **Transportation**: Design road/railway networks

### 2. Clustering
- **Data Mining**: Group similar data points
- **Image Segmentation**: Group similar pixels
- **Bioinformatics**: Cluster similar sequences

### 3. Approximation Algorithms
- **Traveling Salesman**: Approximate TSP using MST
- **Steiner Tree**: Approximate Steiner tree problem
- **Facility Location**: Optimize facility placement

### 4. Graph Theory
- **Connectivity**: Ensure graph remains connected
- **Optimization**: Minimize total connection cost
- **Analysis**: Understand graph structure

## Performance Analysis

The implementation includes performance comparison:

```python
from minimum_spanning_tree import analyze_performance

metrics = analyze_performance(graph)
print(f"Number of vertices: {metrics['num_vertices']}")
print(f"Number of edges: {metrics['num_edges']}")
print(f"Kruskal time: {metrics['kruskal_time']:.6f}s")
print(f"Prim time: {metrics['prim_time']:.6f}s")
print(f"MST weight: {metrics['mst_weight']}")
```

## Visualization

The algorithm includes comprehensive visualization:

```python
from minimum_spanning_tree import visualize_mst

# Visualize graph and MST
visualize_mst(graph, mst_edges, algorithm="Kruskal", show_plot=True)
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

### Kruskal's Implementation
```python
def kruskal_mst(graph):
    """Find MST using Kruskal's algorithm."""
    edges = []
    for u in graph:
        for v, weight in graph[u]:
            if u < v:  # Avoid duplicate edges
                edges.append((weight, u, v))
    
    edges.sort()  # Sort by weight
    mst = []
    uf = UnionFind(len(graph))
    
    for weight, u, v in edges:
        if uf.find(u) != uf.find(v):
            uf.union(u, v)
            mst.append((u, v, weight))
            if len(mst) == len(graph) - 1:
                break
    
    return mst
```

### Prim's Implementation
```python
def prim_mst(graph):
    """Find MST using Prim's algorithm."""
    import heapq
    
    mst = []
    visited = set()
    pq = [(0, 0, -1)]  # (weight, vertex, parent)
    
    while pq and len(visited) < len(graph):
        weight, u, parent = heapq.heappop(pq)
        
        if u in visited:
            continue
            
        visited.add(u)
        if parent != -1:
            mst.append((parent, u, weight))
        
        for v, w in graph[u]:
            if v not in visited:
                heapq.heappush(pq, (w, v, u))
    
    return mst
```

## Edge Cases

### Disconnected Graph
Functions return a minimum spanning forest (one MST per connected component) without emitting warnings.
To detect or warn about disconnected inputs, use the provided `verify_mst` helper or extend these functions with explicit connectivity checks.
### Negative Weights
- Works correctly with negative weights
- MST definition remains valid
- Algorithm behavior unchanged

### Large Graphs
- Memory-efficient implementation
- Handles graphs with millions of vertices
- Optional parallel processing

## Comparison of Algorithms

| Algorithm | Time Complexity | Space Complexity | Best For |
|-----------|----------------|------------------|----------|
| Kruskal's | O(E log E) | O(V) | Sparse graphs |
| Prim's | O(E log V) | O(V) | Dense graphs |
| Borůvka's | O(E log V) | O(V) | Parallel processing |
| Naive | O(V³) | O(V²) | Simple but inefficient |

## Historical Context

Kruskal's algorithm was developed by Joseph Kruskal in 1956, while Prim's algorithm was developed by Robert Prim in 1957. Both algorithms are fundamental in computer science and remain widely used today.

## Applications in Computer Science

### 1. Network Design
- **Internet Routing**: Design efficient network topologies
- **Wireless Networks**: Optimize sensor network placement
- **Distributed Systems**: Design efficient communication networks

### 2. Data Science
- **Clustering**: Group similar data points
- **Dimensionality Reduction**: Find essential connections
- **Feature Selection**: Identify important relationships

### 3. Operations Research
- **Facility Location**: Optimize warehouse placement
- **Supply Chain**: Design efficient distribution networks
- **Transportation**: Plan optimal routes

### 4. Bioinformatics
- **Phylogenetic Trees**: Construct evolutionary trees
- **Protein Networks**: Analyze protein interactions
- **Gene Networks**: Understand gene relationships

## Advanced Topics

### Parallel MST Algorithms
- **Borůvka's Algorithm**: Naturally parallelizable
- **Distributed MST**: Handle very large graphs
- **GPU Implementation**: Accelerate computation

### Dynamic MST
- **Incremental MST**: Add edges efficiently
- **Decremental MST**: Remove edges efficiently
- **Fully Dynamic**: Handle both additions and deletions

### Approximation Algorithms
- **Steiner Tree**: Approximate Steiner tree using MST
- **Traveling Salesman**: Use MST for TSP approximation
- **Facility Location**: MST-based facility placement

### Visualization Features
- **Step-by-step Animation**: Show algorithm progress
- **Interactive Exploration**: Explore different algorithms
- **Performance Comparison**: Compare algorithm efficiency 