class Graph:
    def __init__(self):
        # Initialize an empty graph as a dictionary
        self.graph = {}

    def add_edge(self, u, v):
        # Add an edge from node u to node v
        if u not in self.graph:
            self.graph[u] = []  # Create an empty list for u if not exists
        self.graph[u].append(v)  # Append v to the adjacency list of u

    def dfs(self, start, visited=None):
        # Perform depth-first search starting from the given node
        if visited is None:
            visited = set()  # Initialize visited set if it's the first call
        visited.add(start)  # Mark the current node as visited
        print(start, end=' ')  # Print the current node
        for neighbor in self.graph.get(start, []):
            if neighbor not in visited:
                self.dfs(neighbor, visited)  # Recur for all unvisited neighbors

def menu():
    g = Graph()  # Create a new Graph instance
    while True:
        # Display menu options
        print("\n1. Add Edge\n2. Perform DFS\n3. Exit")
        choice = int(input("Choose an option: "))
        if choice == 1:
            u = input("Enter starting node: ")  # Get starting node
            v = input("Enter ending node: ")  # Get ending node
            g.add_edge(u, v)  # Add edge to the graph
        elif choice == 2:
            start = input("Enter starting node for DFS: ")  # Get starting node for DFS
            print("DFS Traversal: ", end='')
            g.dfs(start)  # Perform DFS from the starting node
        elif choice == 3:
            break  # Exit the loop and program
        else:
            print("Invalid choice!")  # Handle invalid input

if __name__ == "__main__":
    menu()  # Run the menu function if this file is executed
