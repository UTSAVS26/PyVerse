def bellman_ford(graph, V, start):
    """Bellman-Ford algorithm for finding shortest paths from 'start' to all vertices.
       V is the number of vertices, and graph is a list of edges in (u, v, w) format."""
    dist = [float('inf')] * V
    dist[start] = 0

    for _ in range(V - 1):
        for u, v, w in graph:
            if dist[u] != float('inf') and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w

    for u, v, w in graph:
        if dist[u] != float('inf') and dist[u] + w < dist[v]:
            print("Graph contains a negative weight cycle")
            return None

    return dist

def main():
    graph = [
        (0, 1, 5),
        (1, 2, 3),
        (0, 2, 10)
    ]
    V = 3
    start = 0
    print(f"Shortest paths from node {start}: {bellman_ford(graph, V, start)}")

if __name__ == "__main__":
    main()
