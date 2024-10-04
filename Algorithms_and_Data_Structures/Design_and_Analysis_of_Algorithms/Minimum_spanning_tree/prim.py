import sys

class Graph():
    def __init__(self, vertices):
        # Initialize the graph with the given number of vertices
        self.V = vertices
        self.graph = [[0 for _ in range(vertices)] for _ in range(vertices)]  # Adjacency matrix

    def printMST(self, parent):
        # Function to print the constructed MST
        print("Edge \tWeight")
        for i in range(1, self.V):
            # Print each edge and its weight
            print(f"{parent[i]} - {i} \t{self.graph[i][parent[i]]}")

    def minKey(self, key, mstSet):
        # Function to find the vertex with the minimum key value not included in the MST
        min_value = sys.maxsize  # Initialize minimum value to infinity
        min_index = 0  # Initialize index of the minimum key
        for v in range(self.V):
            # Update min_value and min_index if a smaller key is found
            if key[v] < min_value and not mstSet[v]:
                min_value = key[v]
                min_index = v
        return min_index

    def primMST(self):
        # Function to construct the MST using Prim's algorithm
        key = [sys.maxsize] * self.V  # Initialize all keys to infinity
        parent = [None] * self.V  # Array to store the constructed MST
        key[0] = 0  # Make the first vertex the root
        mstSet = [False] * self.V  # To keep track of vertices included in the MST
        parent[0] = -1  # First node is the root of the MST

        # The loop runs V-1 times to construct the MST
        for _ in range(self.V):
            # Get the vertex with the minimum key value
            u = self.minKey(key, mstSet)
            mstSet[u] = True  # Include the vertex in the MST

            # Update the key value and parent index of the adjacent vertices
            for v in range(self.V):
                # Only update the key if the edge weight is less than the current key value
                if self.graph[u][v] > 0 and not mstSet[v] and key[v] > self.graph[u][v]:
                    key[v] = self.graph[u][v]
                    parent[v] = u  # Update parent to the current vertex

        self.printMST(parent)  # Print the resulting MST

if __name__ == '__main__':
    vertices = int(input("Enter the number of vertices: "))  # Input number of vertices
    g = Graph(vertices)  # Create a new graph instance
    print("Enter the adjacency matrix:")  # Prompt for the adjacency matrix
    for i in range(vertices):
        row = list(map(int, input().split()))  # Input each row of the adjacency matrix
        g.graph[i] = row  # Set the row in the graph

    g.primMST()  # Call the Prim's algorithm function to find the MST
