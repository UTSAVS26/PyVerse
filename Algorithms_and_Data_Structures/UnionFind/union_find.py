"""
Union-Find (Disjoint Set Union, DSU) Data Structure

Supports efficient union and find operations to manage disjoint sets.
Optimized with path compression and union by rank.

Time Complexity (amortized):
- Find: O(α(n)), where α is the inverse Ackermann function (very slow-growing)
- Union: O(α(n))
"""

class UnionFind:
    def __init__(self, size):
        """
        Initialize Union-Find with `size` elements (0-indexed).

        Args:
            size (int): Number of elements.
        """
        if size <= 0:
            raise ValueError("Size must be positive.")
        self.parent = [i for i in range(size)]
        self.rank = [0] * size
        self.count = size  # Number of disjoint sets

    def find(self, x):
        """
        Find the root of element x with path compression.

        Args:
            x (int): Element to find.

        Returns:
            int: Root of x.
        """
        if x != self.parent[x]:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        """
        Merge the sets containing x and y.

        Args:
            x (int): First element.
            y (int): Second element.

        Returns:
            bool: True if union was performed, False if already in the same set.
        """
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return False  # Already connected

        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
        self.count -= 1
        return True

    def connected(self, x, y):
        """
        Check if x and y are in the same set.

        Args:
            x (int): First element.
            y (int): Second element.

        Returns:
            bool: True if connected, False otherwise.
        """
        return self.find(x) == self.find(y)

    def set_count(self):
        """
        Get the current number of disjoint sets.

        Returns:
            int: Number of sets.
        """
        return self.count

    def get_groups(self):
        """
        Return all groups as a list of sets.

        Returns:
            List[Set[int]]: List of groups.
        """
        from collections import defaultdict
        groups = defaultdict(set)
        for i in range(len(self.parent)):
            groups[self.find(i)].add(i)
        return list(groups.values())

    def __str__(self):
        return f"UnionFind({self.parent})"

    def __repr__(self):
        return f"UnionFind(size={len(self.parent)}, sets={self.set_count()})"
