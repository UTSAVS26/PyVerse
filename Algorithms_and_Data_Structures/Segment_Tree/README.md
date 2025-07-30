# Segment Tree

A Segment Tree is a powerful binary tree data structure for efficient range queries and updates on arrays.

## Features

- **Build**: Constructed in O(n) time from an input array.
- **Range Queries**: Efficiently compute the sum of any subarray in O(log n) time.
- **Point Updates**: Update any element in O(log n) time.
- **Memory Efficient**: Uses 2n space for n elements.

## Supported Operations

- `query(left, right)`: Returns the sum of elements in the range [left, right].
- `update(index, value)`: Updates the element at position `index` to `value`.

## Time Complexity

| Operation | Complexity |
|-----------|------------|
| Build     | O(n)       |
| Query     | O(log n)   |
| Update    | O(log n)   |

## Usage

from segment_tree import SegmentTree

arr = [2, = SegmentTree(arr)

print(st.query(1, 4)) # Output: 24 (4 + 5 + 7 + 8)
st.update(2, 10) # arr = 10
print(st.query(0, 2)) # Output: 16 (2 + 4 + 10)

## Example

See `segment_tree_example.py` for more examples and performance demos.

## Testing

Run the test suite:

python test_segment_tree.py


## Extensions

- You can modify this implementation to support range min/max queries, range updates (with lazy propagation), or other associative operations.
- For range minimum queries, replace `+` with `min` in the build, update, and query methods.

## References

- [Segment Tree - GeeksforGeeks](https://www.geeksforgeeks.org/segment-tree-data-structure/)
- [Introduction to Algorithms, CLRS](https://mitpress.mit.edu/books/introduction-algorithms-third-edition)
