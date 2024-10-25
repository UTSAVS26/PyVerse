class DSU:
    def __init__(self, n, union_by_rank=True):
        """
        Initializes DSU with `n` nodes.
        If union_by_rank is True, use union by rank.
        If union_by_rank is False, use union by size.
        """
        self.parent = list(range(n))
        self.rank = [0] * n  # For union by rank
        self.size = [1] * n  # For union by size
        self.union_by_rank = union_by_rank

    def find(self, x):
        """Finds the representative of the set containing `x`."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        """
        Unites the sets containing `x` and `y`.
        Uses union by rank if union_by_rank is True, else uses union by size.
        """
        rootX = self.find(x)
        rootY = self.find(y)

        if rootX != rootY:
            if self.union_by_rank:
                # Union by rank
                if self.rank[rootX] > self.rank[rootY]:
                    self.parent[rootY] = rootX
                elif self.rank[rootX] < self.rank[rootY]:
                    self.parent[rootX] = rootY
                else:
                    self.parent[rootY] = rootX
                    self.rank[rootX] += 1
            else:
                # Union by size
                if self.size[rootX] > self.size[rootY]:
                    self.parent[rootY] = rootX
                    self.size[rootX] += self.size[rootY]
                else:
                    self.parent[rootX] = rootY
                    self.size[rootY] += self.size[rootX]


def main():
    # Initialize DSU with union by rank
    print("Union by Rank:")
    dsu_rank = DSU(5, union_by_rank=True)
    dsu_rank.union(0, 1)
    dsu_rank.union(1, 2)
    print("Representative of node 2:", dsu_rank.find(2))
    print("Representative of node 0:", dsu_rank.find(0))

    # Initialize DSU with union by size
    print("\nUnion by Size:")
    dsu_size = DSU(5, union_by_rank=False)
    dsu_size.union(0, 1)
    dsu_size.union(1, 2)
    print("Representative of node 2:", dsu_size.find(2))
    print("Representative of node 0:", dsu_size.find(0))


if __name__ == "__main__":
    main()
