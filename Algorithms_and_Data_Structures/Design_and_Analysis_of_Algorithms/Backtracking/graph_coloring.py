# Number of vertices in the graph
V = 4

def print_solution(color):
    # Print the solution if it exists
    print("Solution Exists: Following are the assigned colors")
    print(" ".join(map(str, color)))  # Print colors assigned to each vertex

def is_safe(v, graph, color, c):
    # Check if it is safe to assign color c to vertex v
    for i in range(V):
        # If there is an edge between v and i, and i has the same color, return False
        if graph[v][i] and c == color[i]:
            return False
    return True  # Color assignment is safe

def graph_coloring_util(graph, m, color, v):
    # Base case: If all vertices are assigned a color
    if v == V:
        return True

    # Try different colors for vertex v
    for c in range(1, m + 1):
        # Check if assigning color c to vertex v is safe
        if is_safe(v, graph, color, c):
            color[v] = c  # Assign color c to vertex v

            # Recur to assign colors to the next vertex
            if graph_coloring_util(graph, m, color, v + 1):
                return True  # If successful, return True

            color[v] = 0  # Backtrack: remove color c from vertex v

    return False  # If no color can be assigned, return False

def graph_coloring(graph, m):
    # Initialize color assignment for vertices
    color = [0] * V

    # Start graph coloring utility function
    if not graph_coloring_util(graph, m, color, 0):
        print("Solution does not exist")  # If no solution exists
        return False

    print_solution(color)  # Print the colors assigned to vertices
    return True  # Solution found

def main():
    print("Enter the number of vertices:")
    global V  # Declare V as global to modify it
    V = int(input())  # Read the number of vertices from user
    
    graph = []  # Initialize an empty list for the adjacency matrix
    print("Enter the adjacency matrix (0 for no edge, 1 for edge):")
    for _ in range(V):
        row = list(map(int, input().split()))  # Read each row of the adjacency matrix
        graph.append(row)  # Append the row to the graph

    m = int(input("Enter the number of colors: "))  # Read the number of colors from user
    
    graph_coloring(graph, m)  # Call the graph coloring function

if __name__ == "__main__":
    main()  # Run the main function
