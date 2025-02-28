class Graph(): 
    def __init__(self, vertices): 
        # Initialize the adjacency matrix for the graph with all zeros
        self.graph = [[0 for _ in range(vertices)] for _ in range(vertices)] 
        self.V = vertices  # Store the number of vertices

    def is_safe(self, v, pos, path): 
        # Check if the vertex v can be added to the Hamiltonian Cycle
        # It should be adjacent to the last vertex in the path and not already in the path
        if self.graph[path[pos-1]][v] == 0: 
            return False  # Not adjacent
        return v not in path[:pos]  # Check if v is already in the path

    def ham_cycle_util(self, path, pos): 
        # Base case: if all vertices are included in the path
        if pos == self.V: 
            # Check if there is an edge from the last vertex to the first vertex
            return self.graph[path[pos-1]][path[0]] == 1

        # Try different vertices as the next candidate in the Hamiltonian Cycle
        for v in range(1, self.V): 
            if self.is_safe(v, pos, path):  # Check if adding vertex v is safe
                path[pos] = v  # Add vertex v to the path

                # Recur to construct the rest of the path
                if self.ham_cycle_util(path, pos + 1): 
                    return True  # If successful, return True
                
                path[pos] = -1  # Backtrack: remove vertex v from the path

        return False  # No Hamiltonian Cycle found

    def ham_cycle(self): 
        path = [-1] * self.V  # Initialize path array
        path[0] = 0  # Start at the first vertex (0)
        
        # Start the utility function to find the Hamiltonian Cycle
        if not self.ham_cycle_util(path, 1): 
            print("Solution does not exist\n")  # If no cycle exists
            return False
        
        self.print_solution(path)  # Print the solution if found
        return True

    def print_solution(self, path): 
        # Print the Hamiltonian Cycle
        print("Solution Exists: Following is one Hamiltonian Cycle")
        print(" -> ".join(map(str, path + [path[0]])))  # Include the start point to complete the cycle

def main():
    vertices = int(input("Enter the number of vertices: "))  # Get number of vertices from user
    g = Graph(vertices)  # Create a graph object
    print("Enter the adjacency matrix (0 for no edge, 1 for edge):")
    
    # Read the adjacency matrix from user input
    for i in range(vertices):
        row = list(map(int, input().split()))
        g.graph[i] = row  # Assign each row to the graph's adjacency matrix

    g.ham_cycle()  # Start the Hamiltonian Cycle function

if __name__ == "__main__":
    main()  # Run the main function
