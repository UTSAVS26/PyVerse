import heapq


def dijkstra(graph, start):
    """Dijkstra's algorithm for shortest paths from the starting node."""

    dist = {node: float('inf') for node in graph}
    dist[start] = 0
    pq = [(0, start)]

    while pq:
        current_distance, current_node = heapq.heappop(pq)

        if current_distance > dist[current_node]:
            continue

        for neighbor, weight in graph[current_node]:
            distance = current_distance + weight

            if distance < dist[neighbor]:
                dist[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))

    return dist


def main():
    graph = {
        0: [(1, 4), (2, 1)],
        1: [(2, 2), (3, 5)],
        2: [(3, 8)],
        3: []
    }
    start = 0
    print(f"Shortest paths from node {start}: {dijkstra(graph, start)}")


if __name__ == "__main__":
    main()
