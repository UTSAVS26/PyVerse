def DFS(adj, V, s, visited):
    visited[s] = True
    print(s, end=' ')
    for u in adj[s]:
        if visited[u] == False:
            DFS(adj, V, u, visited)
            
            
def main():
    
    adj = {
        0 : [1, 4],
        1 : [0, 2],
        2 : [1, 3],
        3 : [2],
        4 : [0, 5, 6],
        5 : [4, 6],
        6 : [4, 5],
    }
    
    V = len(adj)
    visited = [False] * V
    
    for i in adj.keys():
        if visited[i] == False:
            DFS(adj, V, i, visited)
            print()
    
main()