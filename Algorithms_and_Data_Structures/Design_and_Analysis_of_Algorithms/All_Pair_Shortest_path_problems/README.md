# All-Pairs Shortest Path (APSP)

## What is All-Pairs Shortest Path?

The **All-Pairs Shortest Path (APSP)** problem is a classic problem in graph theory. It involves finding the shortest paths between all pairs of nodes in a weighted graph. For every pair of nodes \(u\) and \(v\), the algorithm determines the shortest distance (or path) from \(u\) to \(v\).

### Problem Statement:

Given a graph \(G\) with \(n\) vertices and weighted edges, find the shortest paths between every pair of vertices. The weights on the edges may be positive or negative, but the graph should not contain negative-weight cycles.

### Key Algorithms:

There are several efficient algorithms designed to solve the APSP problem, two of the most well-known being:

1. **Floyd-Warshall Algorithm**
2. **Johnson’s Algorithm**

---

## 1. Floyd-Warshall Algorithm

### Overview:

The **Floyd-Warshall Algorithm** is a dynamic programming-based algorithm used to solve the APSP problem in \(O(n^3)\) time, where \(n\) is the number of vertices in the graph. It works by iteratively improving the shortest path estimates between all pairs of nodes, considering each node as an intermediate point along potential paths.

### Steps:

1. **Initialization**: Start with a distance matrix where the direct edge weight between vertices is given. If no edge exists, the distance is set to infinity.
2. **Update**: For each pair of nodes, check whether including an intermediate node results in a shorter path. Update the distance matrix accordingly.
3. **Result**: The final matrix contains the shortest paths between all pairs of nodes.

### Time Complexity:

- **Time Complexity**: \(O(n^3)\)
- **Space Complexity**: \(O(n^2)\)

### Applications:

- **Routing Algorithms**: Used in network routing to determine the most efficient paths for data packets between nodes.
- **Game Development**: Helps in finding the shortest paths for characters or elements to travel in a virtual world.
- **Geographic Mapping Systems**: Identifying the quickest travel routes between cities or locations on a map.

---

## 2. Johnson's Algorithm

### Overview:

**Johnson’s Algorithm** is an advanced approach for solving the APSP problem, especially when the graph contains sparse edges (i.e., fewer edges compared to a complete graph). The algorithm modifies the weights of the graph to ensure no negative-weight edges, then applies **Dijkstra’s Algorithm** from each node.

### Steps:

1. **Graph Reweighting**: Use **Bellman-Ford Algorithm** to adjust the edge weights to ensure all weights are non-negative.
2. **Dijkstra’s Algorithm**: Apply Dijkstra’s Algorithm from each vertex to determine the shortest path to all other vertices.
3. **Result**: After reweighting, the shortest paths are calculated efficiently in \(O(n^2 \log n + nm)\) time.

### Time Complexity:

- **Time Complexity**: \(O(n^2 \log n + nm)\) (where \(m\) is the number of edges)
- **Space Complexity**: \(O(n^2)\)

### Applications:

- **Large Sparse Graphs**: Johnson’s algorithm is preferred when the graph is sparse (i.e., has fewer edges compared to vertices), such as in road networks or network topology.
- **Telecommunication Networks**: Used to optimize routing paths in large-scale communication systems.
- **Social Networks**: Helps in identifying the shortest relationships or interactions between individuals in a social network.

---

## Key Differences Between Floyd-Warshall and Johnson's Algorithm:

| Algorithm         | Time Complexity          | Space Complexity | Suitable For                          |
|-------------------|--------------------------|------------------|---------------------------------------|
| **Floyd-Warshall**| \(O(n^3)\)               | \(O(n^2)\)       | Dense graphs, simpler implementation  |
| **Johnson’s**     | \(O(n^2 \log n + nm)\)   | \(O(n^2)\)       | Sparse graphs, more complex, scalable |

---

## Applications of APSP Algorithms

1. **Network Design**: Efficient pathfinding in telecommunication and transportation networks to ensure minimum-cost routing.
2. **Web Mapping Services**: Shortest path algorithms are used in GPS and web services like Google Maps to find optimal routes between locations.
3. **Social Network Analysis**: Determine centrality, influence, and shortest interactions in social graphs.
4. **Robotics and Pathfinding**: Autonomous systems use APSP algorithms to navigate efficiently in environments by planning the shortest routes between points.
5. **Traffic Management Systems**: In urban settings, APSP helps model and manage traffic flow by finding the least congested routes between intersections.

---

## Conclusion

The **All-Pairs Shortest Path (APSP)** problem is fundamental in graph theory and computer science. Understanding the Floyd-Warshall and Johnson’s algorithms, along with their applications, is critical for solving pathfinding problems in diverse areas like networks, robotics, and optimization.

By mastering these algorithms, you can optimize real-world systems and solve complex graph-related challenges efficiently.

---
