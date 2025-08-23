# Link-Cut Tree

## Overview
A Link-Cut Tree is a data structure that maintains a forest of rooted trees and supports dynamic tree operations like link, cut, and path queries. It's particularly useful for dynamic connectivity problems.

## Theory

### What is a Link-Cut Tree?
A link-cut tree is a data structure that represents a forest of rooted trees and supports efficient dynamic operations. It uses splay trees to maintain preferred paths and provides amortized logarithmic time for most operations.

### Key Properties
- **Dynamic Operations**: Supports link, cut, and path operations
- **Amortized Performance**: O(log n) amortized time for most operations
- **Path Operations**: Efficient path queries and updates
- **Forest Management**: Maintains multiple trees simultaneously

### Structure
- **Preferred Paths**: Paths that are splayed to root
- **Splay Trees**: Used to maintain preferred paths
- **Path Parents**: Pointers to parent paths
- **Virtual Trees**: Trees of preferred paths

## Applications

### 1. Dynamic Connectivity
- Network connectivity problems
- Graph algorithms
- Online algorithms

### 2. Network Routing
- Dynamic routing tables
- Load balancing
- Traffic optimization

### 3. Game Theory
- Dynamic game trees
- Decision tree analysis
- Strategy optimization

## Algorithm Implementation

### Core Functions

#### `link(link_cut_tree: LinkCutTree, u: int, v: int) -> bool`
Links two trees by making v a child of u.

**Time Complexity**: O(log n) amortized
**Space Complexity**: O(1)

#### `cut(link_cut_tree: LinkCutTree, u: int, v: int) -> bool`
Cuts the edge between u and v.

**Time Complexity**: O(log n) amortized
**Space Complexity**: O(1)

#### `find_root(link_cut_tree: LinkCutTree, u: int) -> int`
Finds the root of the tree containing u.

**Time Complexity**: O(log n) amortized
**Space Complexity**: O(1)

#### `path_query(link_cut_tree: LinkCutTree, u: int, v: int) -> int`
Performs a query on the path from u to v.

**Time Complexity**: O(log n) amortized
**Space Complexity**: O(1)

## Usage Examples

### Basic Operations
```python
# Create link-cut tree
link_cut_tree = LinkCutTree()

# Link nodes
link(link_cut_tree, 1, 2)
link(link_cut_tree, 2, 3)

# Find root
root = find_root(link_cut_tree, 3)
# Returns 1 (root of the tree)
```

### Path Operations
```python
# Path query
result = path_query(link_cut_tree, 1, 3)
# Returns result of query on path from 1 to 3

# Cut edge
cut(link_cut_tree, 2, 3)
# Cuts the edge between nodes 2 and 3
```

## Advanced Features

### 1. Multiple Operations
- Link and cut operations
- Path queries and updates
- Connectivity queries

### 2. Performance Analysis
- Operation time analysis
- Amortized cost analysis
- Memory usage analysis

### 3. Advanced Operations
- Range path queries
- Dynamic edge weights
- Tree decomposition

## Performance Analysis

### Amortized Analysis
- **Link**: O(log n) amortized
- **Cut**: O(log n) amortized
- **Find Root**: O(log n) amortized
- **Path Query**: O(log n) amortized

### Space Usage
- **Node Structure**: O(n) total nodes
- **Splay Trees**: O(n) total space
- **Overall**: O(n) space complexity

### Dynamic Behavior
- **Adaptive**: Performance improves with repeated access
- **Locality**: Recently accessed paths are faster
- **Balancing**: Automatic tree balancing

## Visualization

The implementation includes visualization capabilities:
- **Forest Structure**: Visual representation of the forest
- **Preferred Paths**: Highlighting of preferred paths
- **Operation Process**: Step-by-step operation visualization
- **Performance Metrics**: Operation time graphs

## Test Cases

### 1. Basic Functionality
- Link and cut operations
- Root finding
- Path queries

### 2. Performance Tests
- Large forest processing
- Multiple operation analysis
- Memory usage analysis

### 3. Dynamic Tests
- Dynamic connectivity
- Path updates
- Tree restructuring

## Dependencies
- `typing`: For type hints
- `matplotlib`: For visualization
- `time`: For performance measurement
- `numpy`: For numerical operations

## File Structure
```
Link_Cut_Tree/
├── README.md
└── link_cut_tree.py
```

## Complexity Summary
| Operation | Amortized Time | Space |
|-----------|----------------|-------|
| Link | O(log n) | O(1) |
| Cut | O(log n) | O(1) |
| Find Root | O(log n) | O(1) |
| Path Query | O(log n) | O(1) |

## References
- [Link-Cut Tree - Wikipedia](https://en.wikipedia.org/wiki/Link/cut_tree)
- [Self-Adjusting Binary Search Trees](https://www.cs.cmu.edu/~sleator/papers/self-adjusting.pdf)
- [Applications in Dynamic Connectivity](https://dl.acm.org/doi/10.1145/800141.804678) 