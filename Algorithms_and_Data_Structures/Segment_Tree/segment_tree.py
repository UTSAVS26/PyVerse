"""
Segment Tree Data Structure

A Segment Tree is a binary tree used for efficient range queries and updates
on arrays. Common operations include range sum, range min/max, and point updates.

Time Complexity:
- Build: O(n)
- Query: O(log n)
- Update: O(log n)

This implementation supports range sum queries and point updates.
"""

class SegmentTree:
    def __init__(self, data):
        """
        Initialize the Segment Tree with an input array.

        Args:
            data (List[int]): The input array to build the segment tree from.
        """
        if not data:
            raise ValueError("Input array must not be empty.")
        self.n = len(data)
        self.tree = [0] * (2 * self.n)
        # Build the tree
        for i in range(self.n):
            self.tree[self.n + i] = data[i]
        for i in range(self.n - 1, 0, -1):
            self.tree[i] = self.tree[2 * i] + self.tree[2 * i + 1]

    def update(self, index, value):
        """
        Update the element at index to value.

        Args:
            index (int): The index to update.
            value (int): The new value.
        """
        if not (0 <= index < self.n):
            raise IndexError("Index out of bounds.")
        pos = index + self.n
        self.tree[pos] = value
        while pos > 1:
            pos //= 2
            self.tree[pos] = self.tree[2 * pos] + self.tree[2 * pos + 1]

    def query(self, left, right):
        """
        Compute the sum of elements in the range [left, right].

        Args:
            left (int): Left index (inclusive).
            right (int): Right index (inclusive).

        Returns:
            int: The sum in the range [left, right].
        """
        if not (0 <= left <= right < self.n):
            raise IndexError("Query indices out of bounds.")
        left += self.n
        right += self.n
        result = 0
        while left <= right:
            if left % 2 == 1:
                result += self.tree[left]
                left += 1
            if right % 2 == 0:
                result += self.tree[right]
                right -= 1
            left //= 2
            right //= 2
        return result

    def __str__(self):
        """String representation of the segment tree (for debugging)."""
        return f"SegmentTree({self.tree[self.n:self.n + self.n]})"

    def __repr__(self):
        return f"SegmentTree(size={self.n})"
