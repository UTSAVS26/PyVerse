def floyd_warshall(graph, V):
    """Floyd-Warshall algorithm for finding shortest paths between all pairs of nodes."""

    dist = [[float('inf')] * V for _ in range(V)]

    for u in range(V):
        dist[u][u] = 0
    for u, v, w in graph:
        dist[u][v] = w

    for k in range(V):
        for i in range(V):
            for j in range(V):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

    return dist


def main():
    V = 4
    graph = [
        (0, 1, 3),
        (1, 2, 1),
        (2, 3, 1),
        (3, 0, 1)
    ]
    print("All-pairs shortest paths:")
    dist = floyd_warshall(graph, V)
    for row in dist:
        print(row)


if __name__ == "__main__":
    main()
