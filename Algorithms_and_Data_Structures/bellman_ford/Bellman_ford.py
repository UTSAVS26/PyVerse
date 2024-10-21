def BellmanFord(src, V, graph):
    '''
    this algorithm works exactly like dijkistra algorthm except it 
    can detect negative edge cycle in the graph

    this algo is little slower than dijkistra algortihm 
    '''
    #src is source vertex   V is no of vertex  graph is graph in the form of a list of list
    dist = [float("Inf")]*V
    dist[src] = 0
    
    for _ in range(V-1):
        for u,v,w in graph:
            if dist[u] != float("Inf") and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w

    
    for u,v,w in graph:
        if dist[u] != float("Inf") and dist[u] + w < dist[v]:
            return -1

    return dist

def printArr(dist):
    V = len(dist)
    print("vertex distance from source")
    for i in range(V):
        print(f"{i}\t\t{dist[i]}")

#example usage of bellman ford algorithm
def main():
    v = 5
    graph = []              
    graph.append([0,1,-1])  #[u=initial vertex, final vertex, w = weight of edge btw them]
    graph.append([0,2,4])
    graph.append([1,2,3])
    graph.append([1,3,2])
    graph.append([1,4,2])
    graph.append([3,2,5])
    graph.append([3,1,1])
    graph.append([4,3,-3])

    dist = BellmanFord(0,v,graph)
    printArr(dist)                  

if __name__ == "__main__":
    main()

'''output
vertex distance from source
0               0 
1               -1
2               2 
3               -2
4               1
'''
