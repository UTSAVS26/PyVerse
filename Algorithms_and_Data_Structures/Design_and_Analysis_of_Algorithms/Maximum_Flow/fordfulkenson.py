from collections import deque

def bfs(rGraph, s, t, parent):
    visited = [False] * len(rGraph)
    q = deque()
    q.append(s)
    visited[s] = True
    parent[s] = -1

    while q:
        u = q.popleft()
        for v in range(len(rGraph)):
            if not visited[v] and rGraph[u][v] > 0:
                q.append(v)
                parent[v] = u
                visited[v] = True

    return visited[t]

def fordFulkerson(graph, s, t):
    rGraph = [row[:] for row in graph]
    parent = [-1] * len(graph)
    max_flow = 0

    while bfs(rGraph, s, t, parent):
        path_flow = float('inf')
        v = t
        while v != s:
            u = parent[v]
            path_flow = min(path_flow, rGraph[u][v])
            v = parent[v]

        v = t
        while v != s:
            u = parent[v]
            rGraph[u][v] -= path_flow
            rGraph[v][u] += path_flow
            v = parent[v]

        max_flow += path_flow

    return max_flow

if __name__ == "__main__":
    V = int(input("Enter the number of vertices: "))
    print("Enter the adjacency matrix (use 0 for no connection):")
    graph = []
    for _ in range(V):
        row = list(map(int, input().split()))
        graph.append(row)

    source = int(input("Enter the source vertex (0 to {}): ".format(V-1)))
    sink = int(input("Enter the sink vertex (0 to {}): ".format(V-1)))

    max_flow = fordFulkerson(graph, source, sink)
    print("The maximum possible flow is", max_flow)
