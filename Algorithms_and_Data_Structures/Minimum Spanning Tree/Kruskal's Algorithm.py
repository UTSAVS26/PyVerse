class Graph:
    def __init__(self, vertices):
        self.V = vertices  
        self.graph = []  

    # Function to add an edge to the graph
    def addEdge(self, u, v, w):
        self.graph.append([u, v, w])

    # A function to find the subset of an element i (with path compression)
    def find(self, parent, i):
        if parent[i] != i:
            parent[i] = self.find(parent, parent[i])
        return parent[i]

    # A function that does union of two subsets x and y (uses union by rank)
    def union(self, parent, rank, x, y):
        xroot = self.find(parent, x)
        yroot = self.find(parent, y)

        # Attach smaller rank tree under root of higher rank tree
        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot
        else:
            parent[yroot] = xroot
            rank[xroot] += 1

    # Function to construct MST using Kruskal's algorithm
    def KruskalMST(self):
        result = []  

        # Step 1: Sort all the edges in non-decreasing order of their weight
        self.graph = sorted(self.graph, key=lambda item: item[2])

        parent = []  
        rank = []    

        # Create V subsets with single elements
        for node in range(self.V):
            parent.append(node)
            rank.append(0)

        e = 0  # Number of edges in MST
        i = 0  # Index variable used for sorted edges

        # Number of edges in MST will be V-1
        while e < self.V - 1:

            # Step 2: Pick the smallest edge and increment the index for next iteration
            u, v, w = self.graph[i]
            i += 1

            x = self.find(parent, u)
            y = self.find(parent, v)

            # If including this edge doesn't cause a cycle, include it in result
            if x != y:
                e += 1
                result.append([u, v, w])
                self.union(parent, rank, x, y)

        return result

if __name__ == "__main__":
    # Input number of vertices
    V = int(input("Enter the number of vertices: "))
    g = Graph(V)

    # Input number of edges
    E = int(input("Enter the number of edges: "))

    # Input the edges
    print("Enter the edges in the format 'u v w' where u and v are vertices, and w is the weight:")
    for _ in range(E):
        u, v, w = map(int, input().split())
        g.addEdge(u, v, w)

    # Get the Minimum Spanning Tree (MST) using Kruskal's algorithm
    mst = g.KruskalMST()

    # Output the MST
    print("Edges in the Minimum Spanning Tree:")
    for u, v, weight in mst:
        print(f"{u} -- {v} == {weight}")
