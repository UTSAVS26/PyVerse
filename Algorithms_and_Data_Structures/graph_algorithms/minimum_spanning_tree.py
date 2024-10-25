import heapq

def prim(graph, V):
    """Prim's Algorithm for Minimum Spanning Tree using a priority queue."""
    mst_set = set()
    min_heap = [(0, 0)]  # (weight, vertex)
    mst_cost = 0

    while len(mst_set) < V:
        weight, u = heapq.heappop(min_heap)
        if u in mst_set:
            continue
        mst_set.add(u)
        mst_cost += weight

        for v, w in graph[u]:
            if v not in mst_set:
                heapq.heappush(min_heap, (w, v))

    return mst_cost

class DSU:
    def __init__(self, n):
        """Disjoint Set Union with Union by Rank and Path Compression."""
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            if self.rank[rootX] > self.rank[rootY]:
                self.parent[rootY] = rootX
            elif self.rank[rootX] < self.rank[rootY]:
                self.parent[rootX] = rootY
            else:
                self.parent[rootY] = rootX
                self.rank[rootX] += 1

def kruskal(graph, V):
    """Kruskal's Algorithm for Minimum Spanning Tree using DSU."""
    edges = sorted(graph, key=lambda x: x[2])  # Sort edges by weight
    dsu = DSU(V)
    mst_cost = 0
    mst_edges = 0

    for u, v, weight in edges:
        if dsu.find(u) != dsu.find(v):
            dsu.union(u, v)
            mst_cost += weight
            mst_edges += 1
            if mst_edges == V - 1:
                break

    return mst_cost

def main():
    # Graph represented as adjacency list for Prim's algorithm
    V = 5
    prim_graph = {
        0: [(1, 2), (3, 6)],
        1: [(0, 2), (2, 3), (3, 8), (4, 5)],
        2: [(1, 3), (4, 7)],
        3: [(0, 6), (1, 8)],
        4: [(1, 5), (2, 7)]
    }
    print("Minimum Spanning Tree Cost using Prim's Algorithm:", prim(prim_graph, V))

    # Graph represented as edge list for Kruskal's algorithm
    kruskal_graph = [
        (0, 1, 2), (0, 3, 6),
        (1, 2, 3), (1, 3, 8), (1, 4, 5),
        (2, 4, 7), (3, 4, 9)
    ]
    print("Minimum Spanning Tree Cost using Kruskal's Algorithm:", kruskal(kruskal_graph, V))

if __name__ == "__main__":
    main()
