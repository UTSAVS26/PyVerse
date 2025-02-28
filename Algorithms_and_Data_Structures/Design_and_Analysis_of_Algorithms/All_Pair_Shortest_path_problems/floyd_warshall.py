def floydWarshall(graph):
    # Create a distance matrix initialized with the values from the input graph
    dist = list(map(lambda i: list(map(lambda j: j, i)), graph))

    # Iterate through each vertex as an intermediate point
    for k in range(len(graph)):
        # Iterate through each source vertex
        for i in range(len(graph)):
            # Iterate through each destination vertex
            for j in range(len(graph)):
                # Update the shortest distance between vertices i and j
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

    # Print the final solution showing shortest distances
    printSolution(dist)

def printSolution(dist):
    # Print the header for the distance matrix
    print("Following matrix shows the shortest distances between every pair of vertices:")
    # Iterate through each row in the distance matrix
    for i in range(len(dist)):
        # Iterate through each column in the distance matrix
        for j in range(len(dist)):
            # Check if the distance is infinite (no connection)
            if dist[i][j] == INF:
                print("%7s" % ("INF"), end=" ")  # Print 'INF' if there's no path
            else:
                print("%7d\t" % (dist[i][j]), end=' ')  # Print the distance if it exists
            # Print a newline at the end of each row
            if j == len(dist) - 1:
                print()

if __name__ == "__main__":
    V = int(input("Enter the number of vertices: "))  # Get the number of vertices from the user
    INF = float('inf')  # Define infinity for graph initialization
    
    graph = []  # Initialize an empty list to represent the graph
    print("Enter the graph as an adjacency matrix (use 'INF' for no connection):")
    # Read the adjacency matrix from user input
    for i in range(V):
        row = list(map(lambda x: float('inf') if x == 'INF' else int(x), input().split()))
        graph.append(row)  # Append each row to the graph

    # Call the Floyd-Warshall function with the constructed graph
    floydWarshall(graph)
