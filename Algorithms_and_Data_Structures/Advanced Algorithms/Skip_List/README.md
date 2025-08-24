# Skip List

## Overview
A Skip List is a probabilistic data structure that allows for efficient search, insertion, and deletion operations. It maintains a linked hierarchy of subsequences, each skipping over fewer elements than the previous one.

## Theory

### What is a Skip List?
A skip list is a data structure that allows fast search within an ordered sequence of elements. It maintains a hierarchy of linked lists, where each level skips over a subset of elements from the level below it.

### Key Properties
- **Probabilistic Structure**: Uses randomization to maintain balance
- **Logarithmic Performance**: O(log n) average case for search, insert, delete
- **Simple Implementation**: Easier to implement than balanced trees
- **Space Efficient**: O(n) space complexity

### Structure
- **Multiple Levels**: Each level is a sorted linked list
- **Skip Pointers**: Higher levels skip over elements in lower levels
- **Random Height**: Each node's height is determined probabilistically
- **Sentinel Nodes**: Special nodes at the beginning and end

## Applications

### 1. Database Systems
- Index structures
- Range queries
- Ordered data storage

### 2. Memory-Efficient Ordered Structures
- Alternative to balanced trees
- Cache-friendly data structures
- Embedded systems

### 3. Network Routing
- Packet routing tables
- Load balancing
- Traffic optimization

## Algorithm Implementation

### Core Functions

#### `insert(skip_list: SkipList, key: int, value: Any) -> None`
Inserts a key-value pair into the skip list.

**Time Complexity**: O(log n) average case
**Space Complexity**: O(1)

#### `search(skip_list: SkipList, key: int) -> Optional[Any]`
Searches for a key in the skip list.

**Time Complexity**: O(log n) average case
**Space Complexity**: O(1)

#### `delete(skip_list: SkipList, key: int) -> bool`
Deletes a key from the skip list.

**Time Complexity**: O(log n) average case
**Space Complexity**: O(1)

#### `get_range(skip_list: SkipList, start: int, end: int) -> List[Tuple[int, Any]]`
Gets all key-value pairs in the specified range.

**Time Complexity**: O(log n + k) where k is the number of elements in range
**Space Complexity**: O(k)

## Usage Examples

### Basic Operations
```python
# Create skip list
skip_list = SkipList()

# Insert elements
insert(skip_list, 10, "value1")
insert(skip_list, 5, "value2")
insert(skip_list, 15, "value3")

# Search for element
result = search(skip_list, 5)
# Returns "value2"
```

### Range Queries
```python
# Get all elements in range [5, 15]
elements = get_range(skip_list, 5, 15)
# Returns [(5, "value2"), (10, "value1"), (15, "value3")]
```

## Advanced Features

### 1. Multiple Operations
- Insert, search, delete
- Range queries
- Bulk operations

### 2. Performance Analysis
- Height distribution analysis
- Access pattern analysis
- Memory usage analysis

### 3. Advanced Operations
- Merge operations
- Split operations
- Iterator support

## Performance Analysis

### Average Case Analysis
- **Search**: O(log n) average case
- **Insert**: O(log n) average case
- **Delete**: O(log n) average case
- **Range Query**: O(log n + k) where k is result size

### Height Distribution
- **Expected Height**: O(log n)
- **Maximum Height**: O(log n) with high probability
- **Level Distribution**: Geometric distribution

### Space Usage
- **Node Structure**: O(n) total nodes
- **Level Pointers**: O(n) total pointers
- **Overall**: O(n) space complexity

## Visualization

The implementation includes visualization capabilities:
- **Multi-level Structure**: Visual representation of all levels
- **Search Path**: Highlighting search paths
- **Insertion Process**: Step-by-step insertion visualization
- **Height Distribution**: Histogram of node heights

## Test Cases

### 1. Basic Functionality
- Insertion and deletion
- Search operations
- Range queries

### 2. Performance Tests
- Large dataset processing
- Height distribution analysis
- Memory usage analysis

### 3. Edge Cases
- Empty skip list
- Duplicate keys
- Range edge cases

## Dependencies
- `typing`: For type hints
- `matplotlib`: For visualization
- `time`: For performance measurement
- `numpy`: For numerical operations
- `random`: For probabilistic height generation

## File Structure
```
Skip_List/
├── README.md
└── skip_list.py
```

## Complexity Summary
| Operation | Average Time | Worst Case | Space |
|-----------|--------------|------------|-------|
| Search | O(log n) | O(n) | O(1) |
| Insert | O(log n) | O(n) | O(1) |
| Delete | O(log n) | O(n) | O(1) |
| Range Query | O(log n + k) | O(n) | O(k) |

## References
- [Skip List - Wikipedia](https://en.wikipedia.org/wiki/Skip_list)
- [Skip Lists: A Probabilistic Alternative to Balanced Trees](https://dl.acm.org/doi/10.1145/78973.78977)
- [Applications in Database Systems](https://dl.acm.org/doi/10.1145/800141.804678) 