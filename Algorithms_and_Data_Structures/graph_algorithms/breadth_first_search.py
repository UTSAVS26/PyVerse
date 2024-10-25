from collections import deque


def bfs(graph, start):
    """Performs Breadth-First Search (BFS) on a graph from the starting node."""
    visited = set()
    queue = deque([start])
    visited.add(start)

    while queue:
        node = queue.popleft()
        print(node, end=" ")
        for neighbor in graph[node]:
            if neighbor not in visited:
                queue.append(neighbor)
                visited.add(neighbor)


def main():
    graph = {
        0: [1, 2],
        1: [2],
        2: [0, 3],
        3: []
    }
    print("BFS traversal starting from node 0:")
    bfs(graph, 0)


if __name__ == "__main__":
    main()
