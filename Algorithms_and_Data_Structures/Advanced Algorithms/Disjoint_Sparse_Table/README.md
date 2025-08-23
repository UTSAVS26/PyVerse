# Disjoint Sparse Table

## Overview
A Disjoint Sparse Table is a data structure that allows efficient range queries on static arrays. It supports any associative operation and provides O(1) query time with O(n log n) preprocessing time.

## Theory

### What is a Disjoint Sparse Table?
A Disjoint Sparse Table preprocesses an array to answer range queries in O(1) for any associative operation. Unlike the classical Sparse Table (which relies on overlapping 2^k ranges and idempotent operations), the disjoint variant, for each level k, partitions the array into blocks of size 2^(k+1) and stores:
- suffix aggregates on the left half
- prefix aggregates on the right half
For a query [l, r], choose k = msb(l ^ r) and combine exactly two disjoint entries: table[k][l] and table[k][r].
### Key Properties
- **Static Data**: Works on arrays that don't change after construction
- **Associative Operations**: Supports any associative operation (sum, min, max, gcd, etc.)
- **Fast Queries**: O(1) query time after preprocessing
- **Space Efficient**: O(n log n) space complexity

### Structure
- Multiple Levels: level k uses blocks of size 2^(k+1)
- Disjoint Storage: left-half suffixes and right-half prefixes per block
- Power-of-Two Blocks: blocks double each level
- Disjoint Coverage: any query [l, r] lies across at most one such block and can be answered by combining two disjoint entries
## Applications

### 1. Range Queries
- Range sum queries
- Range minimum/maximum queries
- Range GCD queries
- Range AND/OR queries

### 2. Static Data Analysis
- Statistical analysis
- Data mining
- Pattern recognition

### 3. Competitive Programming
- Fast range queries
- Optimization problems
- Algorithm contests

## Algorithm Implementation

### Core Functions

#### `build_sparse_table(arr: List[int], operation: str = 'sum') -> DisjointSparseTable`
Builds the disjoint sparse table for the given array.

**Time Complexity**: O(n log n)
**Space Complexity**: O(n log n)

#### `query(sparse_table: DisjointSparseTable, left: int, right: int) -> int`
Performs a range query from left to right.

**Time Complexity**: O(1)
**Space Complexity**: O(1)

#### `range_sum(sparse_table: DisjointSparseTable, left: int, right: int) -> int`
Gets the sum of elements in the range [left, right].

**Time Complexity**: O(1)
**Space Complexity**: O(1)

#### `range_min(sparse_table: DisjointSparseTable, left: int, right: int) -> int`
Gets the minimum element in the range [left, right].

**Time Complexity**: O(1)
**Space Complexity**: O(1)

## Usage Examples

### Basic Range Queries
```python
# Create sparse table
arr = [1, 2, 3, 4, 5, 6, 7, 8]
sparse_table = build_sparse_table(arr, 'sum')

# Query range sum
result = query(sparse_table, 2, 5)
# Returns sum of elements from index 2 to 5
```

# Range minimum
sparse_table_min = build_sparse_table(arr, 'min')
min_result = range_min(sparse_table_min, 1, 6)

# Range maximum
sparse_table_max = build_sparse_table(arr, 'max')
max_result = range_max(sparse_table_max, 0, 7)

## Advanced Features

### 1. Multiple Operations
- Sum, minimum, maximum
- GCD, LCM
- Bitwise operations (AND, OR, XOR)
- Custom associative functions

### 2. Performance Analysis
- Construction time analysis
- Query time analysis
- Memory usage analysis

### 3. Advanced Operations
- Range updates (with additional structures)
- 2D sparse tables
- Dynamic programming optimization

## Performance Analysis

### Construction Time
- **Best Case**: O(n log n) - always the same
- **Worst Case**: O(n log n) - always the same
- **Average Case**: O(n log n) - consistent performance

### Query Time
- **All Queries**: O(1) - constant time
- **No Dependencies**: Query time independent of range size
- **Optimal**: Best possible query time for static data

### Space Usage
- **Table Storage**: O(n log n) total entries
- **Level Structure**: O(log n) levels
- **Overall**: O(n log n) space complexity

## Visualization

The implementation includes visualization capabilities:
- **Table Structure**: Visual representation of the sparse table
- **Query Process**: Step-by-step query visualization
- **Range Coverage**: Highlighting covered ranges
- **Performance Metrics**: Construction and query time graphs

## Test Cases

### 1. Basic Functionality
- Simple range queries
- Different operations
- Edge cases

### 2. Performance Tests
- Large array processing
- Multiple query analysis
- Memory usage analysis

### 3. Operation Tests
- Sum queries
- Min/Max queries
- Custom operations

## Dependencies
- `typing`: For type hints
- `matplotlib`: For visualization
- `time`: For performance measurement
- `numpy`: For numerical operations

## File Structure
```
Disjoint_Sparse_Table/
├── README.md
└── disjoint_sparse_table.py
```

## Complexity Summary
| Operation | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| Construction | O(n log n) | O(n log n) |
| Query | O(1) | O(1) |
| Range Sum | O(1) | O(1) |
| Range Min/Max | O(1) | O(1) |

## References
- [Sparse Table - Wikipedia](https://en.wikipedia.org/wiki/Sparse_table)
- [Range Minimum Query](https://cp-algorithms.com/data_structures/sparse-table.html)
- [Applications in Competitive Programming](https://dl.acm.org/doi/10.1145/800141.804678) 