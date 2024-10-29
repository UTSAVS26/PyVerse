INF = 99

def shortestDist(graph, n, source, target):
    dist = [INF] * (n + 1)
    path = [-1] * (n + 1)

    dist[target] = 0
    path[target] = target

    for i in range(target - 1, 0, -1):
        for j in range(i + 1, n + 1):
            if graph[i][j] != INF and dist[i] > graph[i][j] + dist[j]:
                dist[i] = graph[i][j] + dist[j]
                path[i] = j

    if dist[source] == INF:
        print(f"There is no path from node {source} to node {target}")
        return

    print(f"The shortest path distance from node {source} to node {target} is: {dist[source]}")

    print("Path:", source, end="")
    current = source
    while current != target:
        current = path[current]
        print(f" -> {current}", end="")
    print()

def main():
    n = int(input("Enter the number of vertices in the graph: "))

    graph = [[INF] * (n + 1) for _ in range(n + 1)]

    print(f"Enter the adjacency matrix (use {INF} for INF):")
    for i in range(1, n + 1):
        row = list(map(int, input(f"Row {i}: ").split()))
        for j in range(1, n + 1):
            graph[i][j] = row[j - 1]

    source = int(input("Enter the source node: "))
    target = int(input("Enter the target node: "))

    shortestDist(graph, n, source, target)

if __name__ == "__main__":
    main()
