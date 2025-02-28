from collections import deque

def bfs(adj, s):
    q = deque()  # Initialize a queue using deque
    visited = [False] * len(adj)  # Create a visited list to track visited nodes
    visited[s] = True  # Mark the starting node as visited
    q.append(s)  # Add the starting node to the queue

    while q:
        curr = q.popleft()  # Dequeue the front node
        print(curr, end=" ")  # Print the current node
        for x in adj[curr]:  # Iterate through all adjacent nodes
            if not visited[x]:  # If the node is not visited
                visited[x] = True  # Mark it as visited
                q.append(x)  # Enqueue the adjacent node

def add_edge(adj, u, v):
    adj[u].append(v)  # Add edge from u to v
    adj[v].append(u)  # Since it's an undirected graph, add edge from v to u

if __name__ == "__main__":
    V = int(input("Enter the number of vertices: "))  # Input the number of vertices
    adj = [[] for _ in range(V)]  # Initialize an adjacency list for the graph
    E = int(input("Enter the number of edges: "))  # Input the number of edges
    print("Enter each edge in the format 'u v':")
    
    # Input edges and add them to the adjacency list
    for _ in range(E):
        u, v = map(int, input().split())
        add_edge(adj, u, v)
    
    # Input the starting vertex for BFS
    start_vertex = int(input("Enter the starting vertex for BFS (0 to {}): ".format(V - 1)))
    
    # Perform BFS starting from the chosen vertex
    print("BFS starting from vertex {}: ".format(start_vertex))
    bfs(adj, start_vertex)
