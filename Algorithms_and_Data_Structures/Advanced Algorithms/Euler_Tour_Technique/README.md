# Euler Tour Technique

## Overview
The Euler Tour Technique is a method for converting tree problems into array problems by creating a linear representation of the tree. It's particularly useful for subtree queries and updates.

## Theory

### What is Euler Tour Technique?
The Euler Tour Technique creates a linear representation of a tree by traversing the tree in a specific order, visiting each edge twice (once in each direction) and each node multiple times.

### Key Properties
- **Linear Representation**: Converts tree to array
- **Subtree Queries**: Efficient subtree operations
- **Range Queries**: Supports range queries on the tour
- **Updates**: Supports point and range updates

### Structure
- **Euler Tour**: Linear array representing tree traversal
- **Entry/Exit Times**: First and last occurrence of each node
- **Subtree Ranges**: Contiguous ranges for subtrees
- **LCA Support**: Can find Lowest Common Ancestor

## Applications

### 1. Subtree Queries
- Subtree sum queries
- Subtree minimum/maximum
- Subtree updates

### 2. Tree to Array Conversion
- Range queries on trees
- Segment tree applications
- Dynamic programming on trees

### 3. Network Analysis
- Tree-based networks
- Hierarchical data structures
- Graph algorithms

## Algorithm Implementation

### Core Functions

#### `build_euler_tour(tree: Dict[int, List[int]], root: int = 0) -> EulerTour`
Builds the Euler tour for the given tree.

**Time Complexity**: O(n) where n is the number of nodes
**Space Complexity**: O(n)

#### `subtree_query(euler_tour: EulerTour, node: int, operation: str = 'sum') -> int`
Performs a query on the subtree rooted at node.

**Time Complexity**: O(log n)
**Space Complexity**: O(1)

#### `subtree_update(euler_tour: EulerTour, node: int, value: int) -> None`
Updates all nodes in the subtree rooted at node.

**Time Complexity**: O(log n)
**Space Complexity**: O(1)

#### `path_query(euler_tour: EulerTour, u: int, v: int) -> int`
Performs a query on the path from u to v.

**Time Complexity**: O(log n)
**Space Complexity**: O(1)

## Usage Examples

### Basic Subtree Operations
```python
# Build Euler tour
tree = {0: [1, 2], 1: [3, 4], 2: [5], 3: [], 4: [], 5: []}
euler_tour = build_euler_tour(tree)

# Subtree query
result = subtree_query(euler_tour, 1, 'sum')
# Returns sum of all nodes in subtree rooted at 1
```

### Path Queries
```python
# Path query
path_result = path_query(euler_tour, 1, 5)
# Returns result of query on path from 1 to 5
```

## Advanced Features

### 1. Multiple Operations
- Subtree queries and updates
- Path queries
- Range operations

### 2. Performance Analysis
- Construction time analysis
- Query time analysis
- Memory usage analysis

### 3. Advanced Operations
- LCA queries
- Range path queries
- Dynamic tree updates

## Performance Analysis

### Construction Time
- **Best Case**: O(n) - linear time construction
- **Worst Case**: O(n) - always linear
- **Average Case**: O(n) - consistent performance

### Query Time
- **Subtree Queries**: O(log n) with segment tree
- **Path Queries**: O(log n) with appropriate data structure
- **Updates**: O(log n) for point updates

### Space Usage
- **Tour Array**: O(n) total elements
- **Entry/Exit Arrays**: O(n) space
- **Overall**: O(n) space complexity

## Visualization

The implementation includes visualization capabilities:
- **Tree Structure**: Visual representation of the original tree
- **Euler Tour**: Linear representation of the tour
- **Subtree Ranges**: Highlighting of subtree ranges
- **Query Process**: Step-by-step query visualization

## Test Cases

### 1. Basic Functionality
- Simple subtree queries
- Path queries
- Tree construction

### 2. Performance Tests
- Large tree processing
- Multiple query analysis
- Memory usage analysis

### 3. Complex Trees
- Deep trees
- Wide trees
- Balanced and unbalanced trees

## Dependencies
- `typing`: For type hints
- `matplotlib`: For visualization
- `time`: For performance measurement
- `numpy`: For numerical operations

## File Structure
```
Euler_Tour_Technique/
├── README.md
└── euler_tour_technique.py
```

## Complexity Summary
| Operation | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| Construction | O(n) | O(n) |
| Subtree Query | O(log n) | O(1) |
| Subtree Update | O(log n) | O(1) |
| Path Query | O(log n) | O(1) |

## References
- [Euler Tour Technique - Wikipedia](https://en.wikipedia.org/wiki/Euler_tour_technique)
- [Tree to Array Conversion](https://cp-algorithms.com/graph/lca.html)
- [Applications in Competitive Programming](https://dl.acm.org/doi/10.1145/800141.804678) 