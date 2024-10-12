
# Minimum Spanning Tree (MST) Algorithms
This project demonstrates three algorithms for finding the Minimum Spanning Tree (MST) in a graph. A Minimum Spanning Tree is a subgraph that connects all vertices with the minimum total edge weight, without any cycles.

### Algorithms included:
- Prim's Algorithm
- Kruskal's Algorithm
- SciPy's Minimum Spanning Tree
## 1. Prim's Algorithm
**Description:** Prim’s algorithm grows the MST one edge at a time. It starts from an arbitrary vertex and adds the smallest edge connecting the growing MST to a new vertex.

**How it works:**
  - Begin with any node as the starting point.
  - Repeatedly add the smallest edge connecting a visited node to an unvisited node.
  - Stop when all vertices are included in the MST.

**Time Complexity:**
- Using an adjacency matrix and a simple priority queue (heap), the time complexity is` O(V²)`, where V is the number of vertices.
- If using an adjacency list and a min-heap, the time complexity is `O(E log V)`, where E is the number of edges.
- Usage: Prim’s algorithm is efficient for dense graphs (many edges). It efficiently adds the next closest vertex to the existing tree.

## 2. Kruskal's Algorithm
**Description:** Kruskal’s algorithm sorts all the edges by weight and builds the MST by adding the smallest edge, ensuring no cycles are formed. It uses a Union-Find data structure to detect cycles.

**How it works:**
- Sort all edges by their weights.
- Pick the smallest edge and add it to the MST if it doesn’t form a cycle.
- Stop when the MST contains `V-1` edges (for a graph with V vertices).
  
**Time Complexity:**
- O(E log E), which simplifies to O(E log V), since sorting the edges takes O(E log E) and each find and union operation in the Union-Find structure takes nearly constant time, `O(log*V)`.
- Usage: Kruskal’s algorithm is efficient for sparse graphs (few edges) and uses sorting and union-find to ensure efficiency.

## 3. Minimum Spanning Tree using SciPy
**Description:** SciPy provides a built-in function to compute the Minimum Spanning Tree using the Compressed Sparse Row (CSR) matrix representation of a graph.

**How it works:**
- The graph is represented as a sparse matrix, where non-zero entries represent edge weights.
- SciPy’s minimum_spanning_tree() function efficiently computes the MST.

**Time Complexity:**
- For sparse graphs, using SciPy’s implementation, the time complexity is typically `O(E log V)`.
- Usage: This method is very efficient when working with large, sparse graphs. The sparse matrix format saves memory and speeds up computation.

**Installation and Requirements**<br>
To run the algorithms, you will need:
- Python 3.x
- `SciPy` for the SciPy Minimum Spanning Tree
  
To install SciPy, use:
``` python
pip install scipy
```
## Conclusion
These three algorithms are fundamental for solving the Minimum Spanning Tree problem, each suited for different graph structures:

1. Prim’s algorithm is ideal for dense graphs.
2. Kruskal’s algorithm works well for sparse graphs.
3. SciPy’s MST function is an efficient option for large, sparse graphs.
