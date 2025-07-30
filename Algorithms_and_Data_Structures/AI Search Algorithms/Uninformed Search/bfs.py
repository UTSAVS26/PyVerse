def BFS(adj, V, s, visited):
    queue = []
    
    queue.append(s);
    visited[s] = True
    
    while queue:
        u = queue[0]
        queue.pop(0)
        print(u)
        for v in adj[u]:
            if visited[v] == False:
                visited[v] = True
                queue.append(v)

def main():
    adj = {
        0 : [1, 2],
        1 : [0, 2, 3],
        2 : [0, 1, 3, 4],
        3 : [1, 2, 4],
        4 : [2, 3]        
    }
    
    V = len(adj)
    visited = [False] * V
    
    for i in range(V):
        if visited[i] == False:
            BFS(adj, V, i, visited)
            print()
    
main()