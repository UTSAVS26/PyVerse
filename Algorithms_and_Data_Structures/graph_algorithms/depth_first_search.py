
def depth_first_search(graph, start, visited=None):
    """Performs Depth-First Search (DFS) on a graph from the starting node."""

    if visited is None:
        visited = set()
    visited.add(start)
    print(start, end=" ")

    for neighbor in graph[start]:
        if neighbor not in visited:
            depth_first_search(graph, neighbor, visited)


def main():
    graph = {
        0: [1, 2],
        1: [2],
        2: [0, 3],
        3: []
    }
    print("DFS traversal starting from node 0:")
    depth_first_search(graph, 0)


if __name__ == "__main__":
    main()
