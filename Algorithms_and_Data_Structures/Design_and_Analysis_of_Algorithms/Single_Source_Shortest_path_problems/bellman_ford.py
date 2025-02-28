class Graph:
    def __init__(self, vertices):
        # Initialize the graph with the given number of vertices
        self.V = vertices
        self.graph = []  # List to store the edges of the graph

    def addEdge(self, u, v, w):
        # Add an edge to the graph represented as a tuple (u, v, weight)
        self.graph.append((u, v, w))

    def printArr(self, dist):
        # Print the distances from the source to each vertex
        print("Vertex Distance from Source")
        for i in range(self.V):
            print(f"{i}\t\t{dist[i]}")

    def BellmanFord(self, src):
        # Function to implement the Bellman-Ford algorithm
        dist = [float("Inf")] * self.V  # Initialize distances to all vertices as infinity
        dist[src] = 0  # Distance to the source vertex is 0

        # Relax all edges |V| - 1 times
        for _ in range(self.V - 1):
            for u, v, w in self.graph:
                # Update distance if a shorter path is found
                if dist[u] != float("Inf") and dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w

        # Check for negative weight cycles
        for u, v, w in self.graph:
            if dist[u] != float("Inf") and dist[u] + w < dist[v]:
                print("Graph contains negative weight cycle")
                return

        # Print the computed distances
        self.printArr(dist)

if __name__ == '__main__':
    vertices = int(input("Enter the number of vertices: "))  # Input the number of vertices
    g = Graph(vertices)  # Create a new graph instance

    edges = int(input("Enter the number of edges: "))  # Input the number of edges
    for _ in range(edges):
        # Input each edge in the format (u v weight)
        u, v, w = map(int, input("Enter edge (u v weight): ").split())
        g.addEdge(u, v, w)  # Add the edge to the graph

    src = int(input("Enter the source vertex: "))  # Input the source vertex for distance calculation
    g.BellmanFord(src)  # Call the Bellman-Ford algorithm to compute distances
