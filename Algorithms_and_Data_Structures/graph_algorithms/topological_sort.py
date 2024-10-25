from collections import defaultdict

def topological_sort(graph, V):
    """Performs Topological Sort on a Directed Acyclic Graph (DAG)."""

    in_degree = [0] * V  # Array to keep track of in-degrees of nodes
    for u in graph:
        for v in graph[u]:
            in_degree[v] += 1

    queue = [node for node in range(V) if in_degree[node] == 0]
    top_order = []

    while queue:
        u = queue.pop(0)
        top_order.append(u)

        for v in graph[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    if len(top_order) == V:
        return top_order
    else:
        print("Graph has at least one cycle and topological sorting is not possible.")
        return None

def main():
    V = 6
    graph = {
        5: [2, 0],
        4: [0, 1],
        3: [1],
        2: [3],
        1: [],
        0: []
    }
    print("Topological Sort of the graph:")
    print(topological_sort(graph, V))

if __name__ == "__main__":
    main()
