# Persistent Data Structures

## Overview

Persistent Data Structures are data structures that preserve their previous versions when modified. Unlike traditional mutable data structures, persistent structures create new versions while keeping the old ones intact. This enables efficient versioning, undo operations, and functional programming paradigms.

## Theory

### Key Concepts

1. **Immutability**: Once created, a version cannot be modified
2. **Structural Sharing**: New versions share unchanged parts with previous versions
3. **Path Copying**: Only the path from root to modified nodes is copied
4. **Version Management**: Each modification creates a new version/root

### Types of Persistence

1. **Full Persistence**: All versions can be accessed and modified
2. **Partial Persistence**: Only the latest version can be modified
3. **Confluent Persistence**: Versions can be merged

### Implementation Strategies

1. **Fat Nodes**: Store all values in each node
2. **Path Copying**: Copy only the path from root to modified nodes
3. **Balanced Trees**: Use balanced structures for efficient operations

## Applications

1. **Version Control Systems**: Git-like functionality
2. **Undo/Redo Systems**: Text editors, graphics software
3. **Functional Programming**: Immutable data structures
4. **Database Systems**: Temporal databases
5. **Game State Management**: Save/load game states
6. **Concurrent Programming**: Thread-safe data structures

## Algorithm Implementation

### Core Functions

```python
class PersistentNode:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right
        self.version = 0

class PersistentTree:
    def __init__(self):
        self.versions = {}  # version -> root
        self.current_version = 0
        self.next_version = 1

def create_version(tree, version_id):
    """Create a new version of the tree"""
    
def insert(tree, value, version=None):
    """Insert value into specified version"""
    
def delete(tree, value, version=None):
    """Delete value from specified version"""
    
def search(tree, value, version=None):
    """Search for value in specified version"""
    
def get_all_versions(tree):
    """Get all available versions"""
    
def compare_versions(tree, version1, version2):
    """Compare two versions of the tree"""
```

## Usage Examples

### Basic Operations

```python
# Create persistent tree
tree = PersistentTree()

# Insert elements
tree.insert(5)
tree.insert(3)
tree.insert(7)

# Create new version
tree.create_version("v1")

# Modify in new version
tree.insert(2, version="v1")

# Search in different versions
result1 = tree.search(2)  # Not found in current version
result2 = tree.search(2, version="v1")  # Found in v1
```

### Advanced Operations

```python
# Version comparison
diff = tree.compare_versions("v1", "v2")

# Undo operations
tree.undo_to_version("v1")

# Branch creation
tree.create_branch("v1", "branch1")
```

## Advanced Features

### 1. Version Management
- Create, delete, and merge versions
- Version tagging and naming
- Version history tracking

### 2. Efficient Storage
- Structural sharing optimization
- Garbage collection for unused versions
- Memory usage analysis

### 3. Advanced Operations
- Range queries across versions
- Bulk operations
- Version merging

### 4. Performance Optimization
- Lazy evaluation
- Caching strategies
- Compression techniques

## Performance Analysis

### Time Complexity
- For the current unbalanced BST:
  - Insert: O(h) (worst-case O(n))
  - Delete: O(h) (worst-case O(n))
  - Search: O(h) (worst-case O(n))
- With a balanced persistent tree (future enhancement), these become O(log n).
- **Version Creation**: O(1)

### Space Complexity
- **Worst Case**: O(n * v) where v is number of versions
- **Average Case**: O(n + m) where m is total modifications
- **Optimized**: O(n + log v) with structural sharing
### Memory Usage
- **Node Sharing**: Reduces memory by sharing unchanged subtrees
- **Version Storage**: Each version stores only modified path
- **Garbage Collection**: Automatic cleanup of unused versions

## Visualization

### Tree Visualization
- Display all versions side by side
- Highlight differences between versions
- Show structural sharing

### Version Graph
- Visualize version relationships
- Show branching and merging
- Track modification history

### Performance Metrics
- Memory usage over time
- Operation costs per version
- Sharing efficiency analysis

## Test Cases

### Basic Functionality
```python
def test_basic_operations():
    tree = PersistentTree()
    
    # Test insertion
    tree.insert(5)
    assert tree.search(5) is not None
    
    # Test versioning
    tree.create_version("v1")
    tree.insert(3, version="v1")
    assert tree.search(3) is None  # Not in current version
    assert tree.search(3, version="v1") is not None  # In v1
```

### Advanced Scenarios
```python
def test_version_management():
    tree = PersistentTree()
    
    # Create multiple versions
    tree.insert(1)
    tree.create_version("v1")
    tree.insert(2, version="v1")
    tree.create_version("v2", base="v1")
    tree.insert(3, version="v2")
    
    # Test version comparison
    diff = tree.compare_versions("v1", "v2")
    assert len(diff) > 0
```

### Edge Cases
```python
def test_edge_cases():
    tree = PersistentTree()
    
    # Empty tree operations
    assert tree.search(1) is None
    
    # Invalid version access
    try:
        tree.search(1, version="invalid")
        assert False
    except ValueError:
        pass
```

## Dependencies

### Required Libraries
```python
import time
import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Any
import matplotlib.pyplot as plt
import networkx as nx
from copy import deepcopy
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
Shivansh/Persistent_Data_Structures/
├── README.md
├── persistent_data_structures.py
└── test_persistent_structures.py
```

## Complexity Summary

| Operation | Time Complexity | Space Complexity | Description |
|-----------|----------------|------------------|-------------|
| Insert | O(log n) | O(log n) | Insert element in version |
| Delete | O(log n) | O(log n) | Delete element from version |
| Search | O(log n) | O(1) | Search in version |
| Version Create | O(1) | O(1) | Create new version |
| Version Compare | O(n) | O(n) | Compare two versions |
| Get All Versions | O(v) | O(v) | List all versions |
| Memory Usage | O(n + m) | - | Total memory for all versions |

Where:
- n = number of nodes in tree
- v = number of versions
- m = total number of modifications

## Applications in Real-World

1. **Git Version Control**: File system versioning
2. **Text Editors**: Undo/redo functionality
3. **Database Systems**: Temporal data management
4. **Game Development**: Save/load game states
5. **Functional Programming**: Immutable data structures
6. **Concurrent Systems**: Thread-safe operations

## Advanced Topics

### 1. Confluent Persistence
- Merging different versions
- Conflict resolution strategies
- Efficient merge algorithms

### 2. Memory Optimization
- Garbage collection strategies
- Compression techniques
- Lazy evaluation

### 3. Concurrent Access
- Thread-safe operations
- Lock-free implementations
- Distributed versioning

### 4. Specialized Structures
- Persistent arrays
- Persistent hash tables
- Persistent graphs

## Implementation Notes

1. **Structural Sharing**: Maximize sharing of unchanged nodes
2. **Path Copying**: Only copy nodes on the path to modification
3. **Version Management**: Efficient version tracking and cleanup
4. **Memory Management**: Automatic garbage collection
5. **Performance Monitoring**: Track memory usage and operation costs

## Future Enhancements

1. **Compression**: Advanced compression for version storage
2. **Distributed**: Support for distributed versioning
3. **Optimization**: Advanced optimization techniques
4. **Specialization**: Domain-specific persistent structures
5. **Integration**: Integration with existing data structures 