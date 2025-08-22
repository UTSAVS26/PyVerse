# Heavy Light Decomposition

## Overview
Heavy Light Decomposition (HLD) is a technique for decomposing a rooted tree into a set of paths that allows efficient path queries and updates. It's particularly useful for solving problems involving tree path operations.

## Theory

### What is Heavy Light Decomposition?
Heavy Light Decomposition is a technique that decomposes a tree into a set of vertex-disjoint paths called "heavy paths" such that any path from the root to a leaf contains at most O(log n) different heavy paths.

### Key Concepts
- **Heavy Edge**: An edge from a parent to its child with the largest subtree size
- **Light Edge**: Any edge that is not heavy
- **Heavy Path**: A path consisting only of heavy edges
- **Light Edge**: An edge that connects different heavy paths

### Properties
- **Decomposition**: Tree is decomposed into O(log n) heavy paths
- **Path Queries**: Any path can be broken into O(log n) heavy path segments
- **Updates**: Point updates and range updates on paths
- **Queries**: Range queries on paths (sum, min, max, etc.)

## Applications

### 1. Tree Path Queries
- Sum of values on a path
- Minimum/maximum value on a path
- XOR of values on a path
- Count of values satisfying a condition

### 2. Network Routing
- Finding optimal paths in networks
- Load balancing in distributed systems
- Network topology analysis

### 3. Hierarchical Data Structures
- File system organization
- Organizational hierarchies
- Decision tree structures

## Algorithm Implementation

### Core Functions

#### `build_hld(tree: Dict[int, List[int]], root: int = 0) -> HLD`
Constructs the Heavy Light Decomposition of the given tree.

**Time Complexity**: O(n) where n is the number of nodes
**Space Complexity**: O(n)

#### `HLD.path_query(u: int, v: int, operation: str = 'sum') -> int`
Performs a query on the path from u to v using the HLD instance.

**Time Complexity**: O(log²n) for most operations
**Space Complexity**: O(1)
#### `path_update(hld: HLD, u: int, v: int, value: int) -> None`
Updates all nodes on the path from u to v.

**Time Complexity**: O(log²n)
**Space Complexity**: O(1)

#### `subtree_query(hld: HLD, node: int, operation: str = 'sum') -> int`
Performs a query on the subtree rooted at node.

**Time Complexity**: O(log n)
**Space Complexity**: O(1)

# Build HLD
tree = {0: [1, 2], 1: [3, 4], 2: [5], 3: [], 4: [], 5: []}
hld = build_hld(tree)

# Query path sum
result = hld.path_query(3, 5, 'sum')
# Returns sum of values on path from node 3 to node 5

### Path Update
```python
# Update all nodes on a path
path_update(hld, 1, 4, 10)
# Adds 10 to all nodes on path from 1 to 4
```

### Subtree Query
```python
# Query subtree
subtree_sum = subtree_query(hld, 1, 'sum')
# Returns sum of all nodes in subtree rooted at node 1
```

## Advanced Features

### 1. Multiple Operations
- Sum, minimum, maximum, XOR
- Custom aggregation functions
- Lazy propagation for range updates

### 2. Dynamic Trees
- Link/cut operations
- Dynamic edge weights
- Real-time updates

### 3. Advanced Queries
- Lowest Common Ancestor (LCA)
- Distance between nodes
- Path statistics

## Performance Analysis

### Construction Time
- **Best Case**: O(n) - linear time construction
- **Worst Case**: O(n) - always linear
- **Average Case**: O(n) - consistent performance

### Query Time
- **Path Queries**: O(log²n) for most operations
- **Subtree Queries**: O(log n)
- **Point Updates**: O(log n)

### Space Usage
- **Tree Structure**: O(n) nodes and edges
- **Segment Trees**: O(n) for each heavy path
- **Overall**: O(n log n) total space

## Visualization

The implementation includes visualization capabilities:
- **Tree Structure**: Visual representation of the tree
- **Heavy Paths**: Highlighting of heavy paths
- **Path Queries**: Visualization of query paths
- **Updates**: Visual representation of updates

## Test Cases

### 1. Basic Functionality
- Simple path queries
- Subtree queries
- Point updates

### 2. Complex Trees
- Deep trees
- Wide trees
- Balanced and unbalanced trees

### 3. Performance Tests
- Large tree processing
- Multiple queries
- Memory usage analysis

## Dependencies
- `typing`: For type hints
- `matplotlib`: For visualization
- `time`: For performance measurement
- `numpy`: For numerical operations

## File Structure
```
Heavy_Light_Decomposition/
├── README.md
└── heavy_light_decomposition.py
```

## Complexity Summary
| Operation | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| Construction | O(n) | O(n log n) |
| Path Query | O(log²n) | O(1) |
| Path Update | O(log²n) | O(1) |
| Subtree Query | O(log n) | O(1) |

## References
- [Heavy Light Decomposition - Wikipedia](https://en.wikipedia.org/wiki/Heavy_path_decomposition)
- [HLD Tutorial](https://cp-algorithms.com/graph/hld.html)
- [Applications in Network Routing](https://dl.acm.org/doi/10.1145/800141.804678) 