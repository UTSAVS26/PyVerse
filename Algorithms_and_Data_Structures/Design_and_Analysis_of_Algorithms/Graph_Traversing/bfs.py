from collections import deque

def bfs(adj, s):
    q = deque()
    visited = [False] * len(adj)
    visited[s] = True
    q.append(s)

    while q:
        curr = q.popleft()
        print(curr, end=" ")
        for x in adj[curr]:
            if not visited[x]:
                visited[x] = True
                q.append(x)

def add_edge(adj, u, v):
    adj[u].append(v)
    adj[v].append(u)

if __name__ == "__main__":
    V = int(input("Enter the number of vertices: "))
    adj = [[] for _ in range(V)]
    E = int(input("Enter the number of edges: "))
    print("Enter each edge in the format 'u v':")
    for _ in range(E):
        u, v = map(int, input().split())
        add_edge(adj, u, v)
    start_vertex = int(input("Enter the starting vertex for BFS (0 to {}): ".format(V - 1)))
    print("BFS starting from vertex {}: ".format(start_vertex))
    bfs(adj, start_vertex)
