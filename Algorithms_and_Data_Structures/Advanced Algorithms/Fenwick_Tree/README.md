# Fenwick Tree (Binary Indexed Tree)

## Overview

The Fenwick Tree, also known as Binary Indexed Tree (BIT), is a data structure that efficiently supports range sum queries and point updates. It is particularly useful when you need to perform frequent updates and range queries on an array. The Fenwick Tree achieves O(log n) time complexity for both point updates and range sum queries.

## Theory

### Key Concepts

1. **Binary Representation**: Uses the binary representation of indices to determine which elements to update
2. **Least Significant Bit (LSB)**: The rightmost set bit in the binary representation
3. **Range Sum**: Calculated by adding values from multiple nodes
4. **Point Update**: Updates propagate to parent nodes based on LSB

### Core Operations

1. **LSB Calculation**: `i & (-i)` gives the least significant bit
2. **Update**: Add value to current position and propagate to parent
3. **Query**: Sum values from multiple positions using LSB

### Mathematical Foundation

- **Update**: `i += i & (-i)` to move to parent
- **Query**: `i -= i & (-i)` to move to previous position
- **Range Sum**: `query(r) - query(l-1)` for range [l, r]

## Applications

1. **Range Sum Queries**: Efficiently calculate sum of elements in a range
2. **Frequency Counting**: Count occurrences of elements in a range
3. **Inversion Count**: Count inversions in an array
4. **Coordinate Compression**: Handle large coordinate values efficiently
5. **Competitive Programming**: Essential for many algorithmic problems
6. **Database Systems**: Efficient range queries on indexed data

## Algorithm Implementation

### Core Functions

```python
class FenwickTree:
    def __init__(self, size):
        self.size = size
        self.tree = [0] * (size + 1)
    
    def update(self, index, value):
        """Add value to element at index"""
        
    def query(self, index):
        """Get sum from index 1 to index"""
        
    def range_query(self, left, right):
        """Get sum from left to right inclusive"""
        
    def get_array(self):
        """Get the original array from the tree"""
        
    def set_value(self, index, value):
        """Set value at index (not add)"""
```

## Usage Examples

### Basic Operations

```python
# Create Fenwick Tree
ft = FenwickTree(10)

# Update elements
ft.update(1, 5)
ft.update(2, 3)
ft.update(3, 7)

# Query range sum
sum_1_3 = ft.range_query(1, 3)  # 15
sum_2_4 = ft.range_query(2, 4)  # 10
```

### Advanced Operations

```python
# Set specific values
ft.set_value(1, 10)
ft.set_value(2, 20)

# Get original array
original_array = ft.get_array()

# Inversion count
inversions = ft.count_inversions([3, 1, 4, 2])
```

## Advanced Features

### 1. Range Updates
- Support for range add operations
- Lazy propagation techniques
- Efficient bulk updates

### 2. 2D Fenwick Tree
- Extension to 2D arrays
- Matrix range queries
- Image processing applications

### 3. Coordinate Compression
- Handle large coordinate values
- Memory efficient representation
- Dynamic coordinate mapping

### 4. Advanced Queries
- Range minimum/maximum queries
- Range gcd/lcm queries
- Custom associative operations

## Performance Analysis

### Time Complexity
- **Construction**: O(n log n)
- **Point Update**: O(log n)
- **Range Query**: O(log n)
- **Range Update**: O(log n) per element

### Space Complexity
- **Storage**: O(n) additional space
- **Memory Overhead**: Minimal compared to segment trees

### Memory Usage
- **Efficient**: Only stores necessary information
- **Cache Friendly**: Good locality of reference
- **Compact**: Minimal memory overhead

## Visualization

### Tree Structure
- Visual representation of the tree
- Show parent-child relationships
- Highlight update paths

### Update Process
- Animate update operations
- Show propagation to parent nodes
- Visualize LSB calculations

### Query Process
- Show query decomposition
- Highlight nodes involved in sum
- Visualize range calculations

## Test Cases

### Basic Functionality
```python
def test_basic_operations():
    ft = FenwickTree(5)
    
    # Test updates and queries
    ft.update(1, 5)
    ft.update(2, 3)
    assert ft.query(2) == 8
    assert ft.range_query(1, 2) == 8
```

### Advanced Scenarios
```python
def test_range_operations():
    ft = FenwickTree(10)
    
    # Test range updates
    ft.range_update(1, 5, 2)
    assert ft.range_query(1, 5) == 10
    
    # Test coordinate compression
    compressed = ft.compress_coordinates([1000, 2000, 3000])
    assert len(compressed) == 3
```

### Edge Cases
```python
def test_edge_cases():
    ft = FenwickTree(1)
    
    # Test single element
    ft.update(1, 5)
    assert ft.query(1) == 5
    
    # Test empty queries
    assert ft.range_query(2, 1) == 0
```

## Dependencies

### Required Libraries
```python
import time
import numpy as np
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import networkx as nx
```

### Optional Dependencies
```python
# For advanced visualizations
import seaborn as sns
from matplotlib.animation import FuncAnimation
```

## File Structure

```
Shivansh/Fenwick_Tree/
├── README.md
├── fenwick_tree.py
└── test_fenwick_tree.py
```

## Complexity Summary

| Operation | Time Complexity | Space Complexity | Description |
|-----------|----------------|------------------|-------------|
| Construction | O(n log n) | O(n) | Build tree from array |
| Point Update | O(log n) | O(1) | Add value to index |
| Range Query | O(log n) | O(1) | Sum from 1 to index |
| Range Update | O(n log n) | O(1) | Add to range |
| Get Array | O(n log n) | O(n) | Reconstruct original array |
| Inversion Count | O(n log n) | O(n) | Count inversions |

## Applications in Real-World

1. **Database Systems**: Efficient range queries on indexed data
2. **Financial Systems**: Portfolio tracking and risk analysis
3. **Image Processing**: 2D range queries for image manipulation
4. **Bioinformatics**: DNA sequence analysis and pattern matching
5. **Competitive Programming**: Essential for many algorithmic problems
6. **Data Analytics**: Efficient aggregation and reporting

## Advanced Topics

### 1. 2D Fenwick Tree
- Extension to matrices
- 2D range queries and updates
- Image processing applications

### 2. Range Updates
- Lazy propagation techniques
- Efficient bulk operations
- Advanced update strategies

### 3. Coordinate Compression
- Handle large coordinate values
- Dynamic coordinate mapping
- Memory efficient representation

### 4. Custom Operations
- Non-additive operations
- Custom associative functions
- Specialized query types

## Implementation Notes

1. **Indexing**: 1-based indexing for easier implementation
2. **LSB Calculation**: Use bitwise operations for efficiency
3. **Memory Management**: Efficient memory usage with minimal overhead
4. **Error Handling**: Robust error checking for invalid indices
5. **Performance Optimization**: Cache-friendly implementation

## Future Enhancements

1. **3D Extension**: Support for 3D arrays and queries
2. **Dynamic Sizing**: Automatic resizing capabilities
3. **Parallel Processing**: Multi-threaded operations
4. **GPU Acceleration**: CUDA/OpenCL implementations
5. **Persistent Version**: Version control for the tree 