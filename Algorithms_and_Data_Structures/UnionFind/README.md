# Union-Find (Disjoint Set Union, DSU)

A fast and memory-efficient data structure to manage disjoint sets, supporting union and find operations with path compression and union by rank.

## Features

- **Union**: Merge two sets in nearly constant time.
- **Find**: Determine the representative (root) of a set in nearly constant time.
- **Connected**: Check if two elements are in the same set.
- **Group Retrieval**: List all current groups/sets.
- **Optimizations**: Path compression and union by rank for very fast operations.

## Time Complexity

| Operation | Amortized Complexity |
|-----------|---------------------|
| Find      | O(α(n))             |
| Union     | O(α(n))             |
| Connected | O(α(n))             |

(α(n) is the inverse Ackermann function, effectively constant for all practical inputs.)

## Usage

from union_find import UnionFind

uf = UnionFind(5) # 0, 1, 2, 3, 4

uf.union(0, 1)
uf.union(1, 2)
print(uf.connected(0, 2)) # True
print(uf.connected(0, 3)) # False

print(uf.set_count()) # Number of disjoint sets
print(uf.get_groups()) # List of all groups

## Applications

- **Cycle detection** in undirected graphs
- **Kruskal's algorithm** for Minimum Spanning Tree
- **Connected components** in networks
- **Image processing** (labeling connected regions)
- **Percolation theory**

See `union_find_example.py` for more.

## Testing

Run the test suite:
python test_union_find.py


## References

- [Disjoint-set data structure (Wikipedia)](https://en.wikipedia.org/wiki/Disjoint-set_data_structure)
- [Competitive Programming Algorithms](https://cp-algorithms.com/data_structures/disjoint_set_union.html)
- [Introduction to Algorithms, CLRS](https://mitpress.mit.edu/books/introduction-algorithms-third-edition)
