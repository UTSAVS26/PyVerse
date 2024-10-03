def dfs_rec(adj, visited, s):
    visited[s] = True
    print(s, end=" ")
    for i in adj[s]:
        if not visited[i]:
            dfs_rec(adj, visited, i)

def dfs(adj, s):
    visited = [False] * len(adj)
    dfs_rec(adj, visited, s)

def add_edge(adj, s, t):
    adj[s].append(t)
    adj[t].append(s)

if __name__ == "__main__":
    V = int(input("Enter the number of vertices: "))
    E = int(input("Enter the number of edges: "))
    adj = [[] for _ in range(V)]

    print("Enter each edge in the format 'u v':")
    for _ in range(E):
        u, v = map(int, input().split())
        add_edge(adj, u, v)

    source = int(input("Enter the starting vertex for DFS (0 to {}): ".format(V-1)))
    print("DFS from source:", source)
    dfs(adj, source)
