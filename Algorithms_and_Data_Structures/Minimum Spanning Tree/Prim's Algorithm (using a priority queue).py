import heapq
from collections import defaultdict

class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = defaultdict(list)

    def addEdge(self, u, v, w):
        # Adding edge (u, v) with weight w
        self.graph[u].append((v, w))
        self.graph[v].append((u, w))

    def PrimMST(self):
        result = []
        # Set to keep track of visited vertices
        visited = set()
        # Priority queue to pick the edge with the smallest weight
        pq = [(0, 0)]  # (weight, vertex)

        while pq:
            # Get the vertex with the smallest weight
            weight, u = heapq.heappop(pq)
            if u in visited:
                continue
            #Mark this vertex as visited
            visited.add(u)
            #Add the vertex and the weight to the result
            result.append((u, weight))

            #Traverse all the adjacent vertices of u
            for v, w in self.graph[u]:
                if v not in visited:
                    #Push adjacent vertices to the priority queue
                    heapq.heappush(pq, (w, v))

        # Returning the MST result
        return result

if __name__ == "__main__":
    # Take number of vertices as input
    V = int(input("Enter the number of vertices in the graph: "))
    
    g = Graph(V)

    # Take number of edges as input
    E = int(input("Enter the number of edges in the graph: "))

    # Input the edges (u, v, w) from the user
    print("Enter the edges in the format 'u,v,w' where u and v are vertices and w is the weight:")
    for _ in range(E):
        u, v, w = map(int, input().split())
        g.addEdge(u, v, w)

    # Get the Minimum Spanning Tree (MST)
    mst = g.PrimMST()

    # Output
    print("Minimum Spanning Tree:", mst)
