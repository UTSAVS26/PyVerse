from collections import defaultdict

# Define a constant for infinity
INT_MAX = float('Inf')

def Min_Distance(dist, visit):
    # Initialize minimum distance and the corresponding vertex
    minimum, minVertex = INT_MAX, -1
    # Iterate through all vertices to find the vertex with the minimum distance
    for vertex in range(len(dist)):
        if minimum > dist[vertex] and not visit[vertex]:  # Check if the vertex is not visited
            minimum, minVertex = dist[vertex], vertex  # Update minimum distance and vertex
    return minVertex  # Return the vertex with the minimum distance

def Dijkstra_Algorithm(graph, Altered_Graph, source):
    tot_vertices = len(graph)  # Total number of vertices in the graph
    sptSet = defaultdict(lambda: False)  # Set to track the shortest path tree
    dist = [INT_MAX] * tot_vertices  # Initialize distances to infinity
    dist[source] = 0  # Distance from source to itself is 0

    # Loop through all vertices
    for _ in range(tot_vertices):
        curVertex = Min_Distance(dist, sptSet)  # Find the vertex with the minimum distance
        sptSet[curVertex] = True  # Mark the vertex as visited

        # Update distances to adjacent vertices
        for vertex in range(tot_vertices):
            # Check for an edge and if the current distance can be improved
            if (not sptSet[vertex] and 
                dist[vertex] > dist[curVertex] + Altered_Graph[curVertex][vertex] and 
                graph[curVertex][vertex] != 0):
                dist[vertex] = dist[curVertex] + Altered_Graph[curVertex][vertex]  # Update the distance

    # Print the final distances from the source vertex
    for vertex in range(tot_vertices):
        print(f'Vertex {vertex}: {dist[vertex]}')  # Output the distance for each vertex

def BellmanFord_Algorithm(edges, graph, tot_vertices):
    # Initialize distances from source to all vertices as infinity
    dist = [INT_MAX] * (tot_vertices + 1)
    dist[tot_vertices] = 0  # Set the distance to the new vertex (source) as 0

    # Add edges from the new source vertex to all other vertices
    for i in range(tot_vertices):
        edges.append([tot_vertices, i, 0])

    # Relax edges repeatedly for the total number of vertices
    for _ in range(tot_vertices):
        for (source, destn, weight) in edges:
            # Update distance if a shorter path is found
            if dist[source] != INT_MAX and dist[source] + weight < dist[destn]:
                dist[destn] = dist[source] + weight

    return dist[0:tot_vertices]  # Return distances to original vertices

def JohnsonAlgorithm(graph):
    edges = []  # Initialize an empty list to store edges
    # Create edges list from the graph
    for i in range(len(graph)):
        for j in range(len(graph[i])):
            if graph[i][j] != 0:  # Check for existing edges
                edges.append([i, j, graph[i][j]])  # Append edge to edges list

    # Get modified weights using the Bellman-Ford algorithm
    Alter_weights = BellmanFord_Algorithm(edges, graph, len(graph))
    # Initialize altered graph with zero weights
    Altered_Graph = [[0 for _ in range(len(graph))] for _ in range(len(graph))]

    # Update the altered graph with modified weights
    for i in range(len(graph)):
        for j in range(len(graph[i])):
            if graph[i][j] != 0:  # Check for existing edges
                Altered_Graph[i][j] = graph[i][j] + Alter_weights[i] - Alter_weights[j]

    print('Modified Graph:', Altered_Graph)  # Output the modified graph

    # Run Dijkstra's algorithm for each vertex as the source
    for source in range(len(graph)):
        print(f'\nShortest Distance with vertex {source} as the source:\n')
        Dijkstra_Algorithm(graph, Altered_Graph, source)  # Call Dijkstra's algorithm

if __name__ == "__main__":
    V = int(input("Enter the number of vertices: "))  # Get number of vertices from user
    graph = []  # Initialize an empty list for the graph
    print("Enter the graph as an adjacency matrix (use 0 for no connection):")
    # Read the adjacency matrix from user input
    for _ in range(V):
        row = list(map(int, input().split()))  # Read a row of the adjacency matrix
        graph.append(row)  # Append the row to the graph

    # Call the Johnson's algorithm with the input graph
    JohnsonAlgorithm(graph)
