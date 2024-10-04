from collections import deque

def bfs(rGraph, s, t, parent):
    # Initialize visited list to track visited vertices
    visited = [False] * len(rGraph)
    q = deque()  # Create a queue for BFS
    q.append(s)  # Start BFS from the source vertex
    visited[s] = True  # Mark source as visited
    parent[s] = -1  # Source has no parent

    # Perform BFS to find an augmenting path
    while q:
        u = q.popleft()  # Get the front vertex
        for v in range(len(rGraph)):  # Check all vertices
            # If not visited and there's remaining capacity
            if not visited[v] and rGraph[u][v] > 0:
                q.append(v)  # Add vertex to queue
                parent[v] = u  # Update parent
                visited[v] = True  # Mark as visited

    # Return whether we reached the sink
    return visited[t]

def fordFulkerson(graph, s, t):
    # Create a residual graph
    rGraph = [row[:] for row in graph]
    parent = [-1] * len(graph)  # Array to store the path
    max_flow = 0  # Initialize max flow

    # While there's an augmenting path in the residual graph
    while bfs(rGraph, s, t, parent):
        path_flow = float('inf')  # Initialize path flow
        v = t

        # Find the maximum flow through the path found
        while v != s:
            u = parent[v]
            path_flow = min(path_flow, rGraph[u][v])  # Find minimum capacity in the path
            v = parent[v]

        # update residual capacities of the edges and reverse edges
        v = t
        while v != s:
            u = parent[v]
            rGraph[u][v] -= path_flow  # Decrease forward edge capacity
            rGraph[v][u] += path_flow  # Increase reverse edge capacity
            v = parent[v]

        max_flow += path_flow  # Add path flow to overall flow

    return max_flow

if __name__ == "__main__":
    V = int(input("Enter the number of vertices: "))  # Input the number of vertices
    print("Enter the adjacency matrix (use 0 for no connection):")
    graph = []
    for _ in range(V):
        row = list(map(int, input().split()))  # Input each row of the adjacency matrix
        graph.append(row)

    source = int(input("Enter the source vertex (0 to {}): ".format(V-1)))  # Input source vertex
    sink = int(input("Enter the sink vertex (0 to {}): ".format(V-1)))  # Input sink vertex

    max_flow = fordFulkerson(graph, source, sink)  # Calculate max flow using Ford-Fulkerson
    print("The maximum possible flow is", max_flow)  # Output the result
