from collections import deque

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

    def bfs(self, start):
        visited = set()
        queue = deque([start])
        traversal = []

        visited.add(start)

        while queue:
            node = queue.popleft()
            traversal.append(node)
            for neighbor in self.graph.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return traversal

    def display(self):
        for node, neighbors in self.graph.items():
            print(f"{node}: {neighbors}")

def menu():
    print("\n--- Graph BFS Menu ---")
    print("1. Add Edge")
    print("2. Perform BFS")
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
            start = input("Enter start node for BFS: ")
            if start in graph.graph:
                print("BFS Traversal:", graph.bfs(start))
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
