# Fibonacci Heap

## Overview

A Fibonacci Heap is an advanced data structure that provides the best amortized time bounds for many operations. It supports insert, extract-min, decrease-key, and delete operations with excellent amortized performance. The Fibonacci Heap is particularly useful for algorithms like Dijkstra's shortest path and Prim's minimum spanning tree.

## Theory

### Key Concepts

1. **Collection of Trees**: Multiple min-heap ordered trees
2. **Root List**: Circular doubly-linked list of tree roots
3. **Marking**: Nodes are marked to track structure changes
4. **Consolidation**: Merge trees of same degree during extract-min

### Core Operations

1. **Insert**: Add new tree to root list
2. **Extract-Min**: Remove minimum and consolidate
3. **Decrease-Key**: Decrease key and cut if necessary
4. **Delete**: Decrease to -∞ then extract-min

### Mathematical Foundation

- **Amortized Insert**: O(1)
- **Amortized Extract-Min**: O(log n)
- **Amortized Decrease-Key**: O(1)
- **Amortized Delete**: O(log n)
- **Potential Function**: Based on marked nodes and tree structure

## Applications

1. **Graph Algorithms**: Dijkstra's, Prim's, Kruskal's algorithms
2. **Network Routing**: Shortest path computations
3. **Task Scheduling**: Priority-based scheduling systems
4. **Event Simulation**: Discrete event simulation
5. **Database Systems**: Query optimization
6. **Game Development**: AI pathfinding algorithms

## Algorithm Implementation

### Core Functions

```python
class FibonacciNode:
    def __init__(self, key, value=None):
        self.key = key
        self.value = value
        self.degree = 0
        self.marked = False
        self.parent = None
        self.child = None
        self.left = None
        self.right = None

class FibonacciHeap:
    def __init__(self):
        self.min_node = None
        self.n = 0  # Number of nodes
        
    def insert(self, key, value=None):
        """Insert new node with key"""
        
    def extract_min(self):
        """Extract minimum key node"""
        
    def decrease_key(self, node, new_key):
        """Decrease key of given node"""
        
    def delete(self, node):
        """Delete given node"""
        
    def union(self, other):
        """Union with another Fibonacci heap"""
        
    def get_min(self):
        """Get minimum key without removing"""
```

## Usage Examples

### Basic Operations

```python
# Create Fibonacci heap
fh = FibonacciHeap()

# Insert elements
fh.insert(5)
fh.insert(3)
fh.insert(7)
fh.insert(1)

# Extract minimum
min_node = fh.extract_min()  # Returns node with key 1

# Decrease key
node = fh.insert(10)
fh.decrease_key(node, 2)

# Delete node
fh.delete(node)
```

### Advanced Operations

```python
# Union two heaps
fh1 = FibonacciHeap()
fh2 = FibonacciHeap()
fh1.insert(1)
fh2.insert(2)
fh1.union(fh2)

# Get statistics
size = fh.get_size()
min_key = fh.get_min()

# Performance analysis
stats = fh.get_statistics()
```

## Advanced Features

### 1. Consolidation Optimization
- Efficient tree merging
- Degree-based consolidation
- Amortized analysis

### 2. Marking Strategy
- Node marking for efficiency
- Cut and cascade operations
- Potential function maintenance

### 3. Union Operations
- Efficient heap merging
- Root list management
- Minimum tracking

### 4. Performance Monitoring
- Operation counting
- Amortized cost analysis
- Memory usage tracking

## Performance Analysis

### Time Complexity
- **Insert**: O(1) amortized
- **Extract-Min**: O(log n) amortized
- **Decrease-Key**: O(1) amortized
- **Delete**: O(log n) amortized
- **Union**: O(1) amortized

### Space Complexity
- **Storage**: O(n) for n nodes
- **Node Overhead**: Constant per node
- **Memory Efficiency**: Good space utilization

### Memory Usage
- **Efficient**: Only stores necessary information
- **Cache Friendly**: Good locality for operations
- **Compact**: Minimal memory overhead

## Visualization

### Heap Structure
- Visual representation of trees
- Show root list connections
- Highlight minimum node

### Consolidation Process
- Animate tree merging
- Show degree-based consolidation
- Visualize structure changes

### Decrease-Key Process
- Show cut operations
- Animate cascading cuts
- Visualize marking strategy

## Test Cases

### Basic Functionality
```python
def test_basic_operations():
    fh = FibonacciHeap()
    
    # Test insertion
    fh.insert(5)
    fh.insert(3)
    min_node = fh.extract_min()
    assert min_node.key == 3
    
    # Test decrease key
    node = fh.insert(10)
    fh.decrease_key(node, 1)
    min_node = fh.extract_min()
    assert min_node.key == 1
```

### Advanced Scenarios
```python
def test_union_operations():
    fh1 = FibonacciHeap()
    fh2 = FibonacciHeap()
    
    fh1.insert(1)
    fh1.insert(3)
    fh2.insert(2)
    fh2.insert(4)
    
    fh1.union(fh2)
    min_node = fh1.extract_min()
    assert min_node.key == 1
```

### Performance Tests
```python
def test_performance():
    import time
    
    fh = FibonacciHeap()
    start_time = time.time()
    
    # Insert test
    for i in range(10000):
        fh.insert(i)
    
    insert_time = time.time() - start_time
    
    # Extract test
    start_time = time.time()
    for _ in range(1000):
        fh.extract_min()
    
    extract_time = time.time() - start_time
    
    assert insert_time < 1.0  # Should be fast
    assert extract_time < 2.0  # Should be reasonable
```

## Dependencies

### Required Libraries
```python
import time
import numpy as np
from typing import Optional, List, Dict, Any
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
```

### Optional Dependencies
```python
# For advanced visualizations
import seaborn as sns
from matplotlib.animation import FuncAnimation
```

## File Structure

```
Shivansh/Fibonacci_Heap/
├── README.md
├── fibonacci_heap.py
└── test_fibonacci_heap.py
```

## Complexity Summary

| Operation | Amortized Time | Worst Case | Space | Description |
|-----------|----------------|------------|-------|-------------|
| Insert | O(1) | O(1) | O(1) | Insert new node |
| Extract-Min | O(log n) | O(n) | O(1) | Remove minimum |
| Decrease-Key | O(1) | O(log n) | O(1) | Decrease node key |
| Delete | O(log n) | O(n) | O(1) | Delete node |
| Union | O(1) | O(1) | O(1) | Merge two heaps |
| Get-Min | O(1) | O(1) | O(1) | Get minimum key |

## Applications in Real-World

1. **Graph Algorithms**: Dijkstra's shortest path
2. **Network Routing**: Internet routing protocols
3. **Task Scheduling**: Operating system schedulers
4. **Event Simulation**: Discrete event simulation
5. **Database Systems**: Query optimization
6. **Game Development**: AI pathfinding

## Advanced Topics

### 1. Lazy Fibonacci Heaps
- Deferred consolidation
- Improved practical performance
- Memory optimization

### 2. Concurrent Fibonacci Heaps
- Thread-safe operations
- Lock-free implementations
- Multi-threaded algorithms

### 3. Persistent Fibonacci Heaps
- Immutable versions
- Version control
- Functional programming support

### 4. Specialized Variants
- Pairing heaps
- Rank-pairing heaps
- Brodal queues

## Implementation Notes

1. **Node Management**: Efficient circular doubly-linked lists
2. **Consolidation**: Degree-based tree merging
3. **Marking Strategy**: Proper cut and cascade operations
4. **Memory Management**: Efficient node allocation
5. **Error Handling**: Robust input validation

## Future Enhancements

1. **GPU Acceleration**: CUDA/OpenCL implementations
2. **Compression**: Memory-efficient representations
3. **Distributed Processing**: Multi-node heaps
4. **Specialized Variants**: Domain-specific optimizations
5. **Integration**: Graph algorithm libraries 