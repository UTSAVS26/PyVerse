class Graph: 
    def __init__(self, vertices): 
        self.V = vertices  # Number of vertices in the graph
        self.graph = []    # List to store the edges

    def add_edge(self, u, v, w): 
        # Function to add an edge to the graph
        self.graph.append([u, v, w])  # Append edge as a list [u, v, weight]

    def find(self, parent, i): 
        # Function to find the parent of an element i
        if parent[i] != i:
            parent[i] = self.find(parent, parent[i])  # Path compression
        return parent[i] 

    def union(self, parent, rank, x, y): 
        # Function to perform union of two subsets x and y
        if rank[x] < rank[y]: 
            parent[x] = y  # Make y the parent of x
        elif rank[x] > rank[y]: 
            parent[y] = x  # Make x the parent of y
        else: 
            parent[y] = x  # Make x the parent of y
            rank[x] += 1   # Increase rank of x

    def kruskal_mst(self): 
        result = []  # To store the resultant MST
        i, e = 0, 0  # Initialize variables for the current edge and the number of edges in MST
        self.graph.sort(key=lambda item: item[2])  # Sort edges based on their weight
        parent = list(range(self.V))  # Create a parent list for union-find
        rank = [0] * self.V  # Create a rank list for union-find

        # Loop until we include V-1 edges in the MST
        while e < self.V - 1: 
            u, v, w = self.graph[i]  # Get the next edge
            i += 1
            x = self.find(parent, u)  # Find the parent of u
            y = self.find(parent, v)  # Find the parent of v
            if x != y:  # If they belong to different sets
                e += 1  # Increase the count of edges in MST
                result.append([u, v, w])  # Add this edge to the result
                self.union(parent, rank, x, y)  # Union the sets

        # Calculate the total cost of the MST
        minimum_cost = sum(weight for _, _, weight in result)
        print("Edges in the constructed MST") 
        for u, v, weight in result: 
            print(f"{u} -- {v} == {weight}")  # Print each edge
        print("Minimum Spanning Tree Cost:", minimum_cost)  # Print the total cost of MST

def main():
    vertices = int(input("Enter the number of vertices: "))  # Input number of vertices
    g = Graph(vertices)  # Create a new graph instance
    edges = int(input("Enter the number of edges: "))  # Input number of edges
    print("Enter each edge in the format 'u v w' where u and v are vertices and w is the weight:")
    
    for _ in range(edges):
        u, v, w = map(int, input().split())  # Input each edge
        g.add_edge(u, v, w)  # Add the edge to the graph

    g.kruskal_mst()  # Find and print the MST

if __name__ == '__main__': 
    main()  # Run the main function
