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
            if dist[i][j] == float('inf'):
                print("%7s" % ("INF"), end=" ")  # Print 'INF' if there's no path
            else:
                print("%7d\t" % (dist[i][j]), end=' ')  # Print the distance if it exists
        print()  # Newline after each row

def validateInput(matrix, V):
    # Ensure the matrix is V x V and contains only numbers or 'INF'
    if len(matrix) != V:
        return False
    for row in matrix:
        if len(row) != V:
            return False
        for elem in row:
            if not (elem.isdigit() or elem.lower() == 'inf'):
                return False
    return True

if __name__ == "__main__":
    try:
        V = int(input("Enter the number of vertices: "))  # Get the number of vertices from the user
        if V <= 0:
            raise ValueError("Number of vertices must be a positive integer.")

        INF = float('inf')  # Define infinity for graph initialization
        
        graph = []  # Initialize an empty list to represent the graph
        print("Enter the graph as an adjacency matrix (use 'INF' for no connection):")
        
        # Read the adjacency matrix from user input
        for i in range(V):
            row = input(f"Row {i + 1}: ").split()
            graph.append([float('inf') if x.lower() == 'inf' else int(x) for x in row])

        # Validate the input graph
        if not validateInput(graph, V):
            raise ValueError("Invalid input! Ensure the matrix is V x V and contains valid numbers or 'INF'.")

        # Call the Floyd-Warshall function with the constructed graph
        floydWarshall(graph)

    except ValueError as ve:
        print(f"Error: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
