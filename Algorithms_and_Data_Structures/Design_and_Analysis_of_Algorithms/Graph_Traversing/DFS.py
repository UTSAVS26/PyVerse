def dfs_rec(adj, visited, s):
    visited[s] = True  # Mark the current node as visited
    print(s, end=" ")  # Print the current node
    for i in adj[s]:  # Loop through all adjacent vertices
        if not visited[i]:  # If the adjacent vertex is not visited
            dfs_rec(adj, visited, i)  # Recursively perform DFS on the adjacent vertex

def dfs(adj, s):
    visited = [False] * len(adj)  # Initialize a visited list
    dfs_rec(adj, visited, s)  # Start DFS from the source vertex

def add_edge(adj, s, t):
    adj[s].append(t)  # Add an edge from s to t
    adj[t].append(s)  # Since it's an undirected graph, add an edge from t to s

if __name__ == "__main__":
    V = int(input("Enter the number of vertices: "))  # Input the number of vertices
    E = int(input("Enter the number of edges: "))  # Input the number of edges
    adj = [[] for _ in range(V)]  # Initialize an adjacency list for the graph

    print("Enter each edge in the format 'u v':")
    # Input edges and add them to the adjacency list
    for _ in range(E):
        u, v = map(int, input().split())
        add_edge(adj, u, v)

    # Input the starting vertex for DFS
    source = int(input("Enter the starting vertex for DFS (0 to {}): ".format(V - 1)))
    
    # Perform DFS starting from the chosen vertex
    print("DFS from source:", source)
    dfs(adj, source)
