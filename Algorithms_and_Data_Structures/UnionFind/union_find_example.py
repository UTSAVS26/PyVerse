"""
Union-Find Usage Examples

Demonstrates basic and advanced usage of the UnionFind class.
"""

from union_find import UnionFind

def basic_demo():
    print("=== Union-Find Basic Demo ===")
    uf = UnionFind(5)  # Elements: 0, 1, 2, 3, 4

    print("Initial sets:", uf.get_groups())
    uf.union(0, 1)
    print("After union(0, 1):", uf.get_groups())
    uf.union(1, 2)
    print("After union(1, 2):", uf.get_groups())
    print("Are 0 and 2 connected?", uf.connected(0, 2))
    print("Are 0 and 3 connected?", uf.connected(0, 3))
    uf.union(3, 4)
    print("After union(3, 4):", uf.get_groups())
    uf.union(2, 3)
    print("After union(2, 3):", uf.get_groups())
    print("Total sets:", uf.set_count())

def application_demo():
    print("\n=== Union-Find Application Demo: Cycle Detection in Graph ===")
    # Detect cycle in an undirected graph
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (2, 4)]
    n = 5
    uf = UnionFind(n)
    has_cycle = False
    for u, v in edges:
        if uf.connected(u, v):
            has_cycle = True
            print(f"Cycle detected when adding edge ({u}, {v})")
            break
        uf.union(u, v)
    if not has_cycle:
        print("No cycle detected.")

def kruskal_demo():
    print("\n=== Union-Find Application Demo: Kruskal's MST (Conceptual) ===")
    # Kruskal's algorithm: sort edges, add if not connected
    edges = [
        (1, 0, 1),  # (weight, u, v)
        (2, 1, 2),
        (3, 0, 2),
        (4, 3, 4)
    ]
    edges.sort()
    n = 5
    uf = UnionFind(n)
    mst_weight = 0
    for w, u, v in edges:
        if not uf.connected(u, v):
            uf.union(u, v)
            mst_weight += w
            print(f"Edge ({u}, {v}) with weight {w} added to MST.")
    print("MST total weight:", mst_weight)

if __name__ == "__main__":
    basic_demo()
    application_demo()
    kruskal_demo()
