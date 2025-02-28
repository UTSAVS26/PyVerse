import sys

class Graph():
    def __init__(self, vertices):
        # Initialize the graph with the number of vertices
        self.V = vertices
        # Create an adjacency matrix initialized to 0
        self.graph = [[0 for _ in range(vertices)] for _ in range(vertices)]

    def printSolution(self, dist):
        # Print the shortest distances from the source vertex
        print("Vertex \tDistance from Source")
        for node in range(self.V):
            print(f"{node} \t {dist[node]}")

    def minDistance(self, dist, sptSet):
        # Find the vertex with the minimum distance from the set of vertices not yet processed
        min_value = sys.maxsize
        min_index = 0
        for u in range(self.V):
            if dist[u] < min_value and not sptSet[u]:
                min_value = dist[u]
                min_index = u
        return min_index

    def dijkstra(self, src):
        # Implement Dijkstra's algorithm
        dist = [sys.maxsize] * self.V  # Initialize distances to all vertices as infinity
        dist[src] = 0  # Distance to the source vertex is 0
        sptSet = [False] * self.V  # To track vertices included in the shortest path tree

        # Loop through all vertices
        for _ in range(self.V):
            # Get the vertex with the minimum distance from the unvisited set
            x = self.minDistance(dist, sptSet)
            sptSet[x] = True  # Mark the vertex as processed

            # Update the distance value of the neighboring vertices of the picked vertex
            for y in range(self.V):
                if self.graph[x][y] > 0 and not sptSet[y] and dist[y] > dist[x] + self.graph[x][y]:
                    dist[y] = dist[x] + self.graph[x][y]

        # Print the calculated shortest distances
        self.printSolution(dist)

if __name__ == "__main__":
    vertices = int(input("Enter the number of vertices: "))  # Input for number of vertices
    g = Graph(vertices)  # Create a new graph instance
    print("Enter the adjacency matrix (use 0 for no direct path):")
    for i in range(vertices):
        # Input the adjacency matrix row by row
        row = list(map(int, input().split()))
        g.graph[i] = row  # Set the row in the graph

    source = int(input("Enter the source vertex: "))  # Input the source vertex
    g.dijkstra(source)  # Call Dijkstra's algorithm to find the shortest paths
