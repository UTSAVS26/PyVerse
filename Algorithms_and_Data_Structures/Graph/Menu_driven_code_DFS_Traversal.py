class Graph:
    def __init__(self):
        self.graph = {}

    def add_edge(self, u, v):
        if u not in self.graph:
            self.graph[u] = []
        if v not in self.graph:
            self.graph[v] = []
        self.graph[u].append(v)
        self.graph[v].append(u)  # For undirected graph

    def dfs(self, start):
        visited = set()
        traversal = []

        def dfs_util(v):
            visited.add(v)
            traversal.append(v)
            for neighbor in self.graph.get(v, []):
                if neighbor not in visited:
                    dfs_util(neighbor)

        dfs_util(start)
        return traversal

    def display(self):
        for node, neighbors in self.graph.items():
            print(f"{node}: {neighbors}")

def menu():
    print("\n--- Graph DFS Menu ---")
    print("1. Add Edge")
    print("2. Perform DFS")
    print("3. Display Graph")
    print("4. Exit")

if __name__ == "__main__":
    graph = Graph()
    
    while True:
        menu()
        choice = input("Enter choice: ")
        
        if choice == '1':
            u = input("Enter first node: ")
            v = input("Enter second node: ")
            graph.add_edge(u, v)
            print(f"Added edge between {u} and {v}.")
        
        elif choice == '2':
            start = input("Enter start node for DFS: ")
            if start in graph.graph:
                print("DFS Traversal:", graph.dfs(start))
            else:
                print(f"Node {start} does not exist in the graph.")
        
        elif choice == '3':
            print("Graph Adjacency List:")
            graph.display()
        
        elif choice == '4':
            print("Exiting...")
            break
        
        else:
            print("Invalid choice, try again.")
