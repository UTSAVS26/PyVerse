class Graph:
    def __init__(self):
        self.graph = {}

    def add_edge(self, u, v):
        if u not in self.graph:
            self.graph[u] = []
        self.graph[u].append(v)

    def dfs(self, start, visited=None):
        if visited is None:
            visited = set()
        visited.add(start)
        print(start, end=' ')
        for neighbor in self.graph.get(start, []):
            if neighbor not in visited:
                self.dfs(neighbor, visited)

def menu():
    g = Graph()
    while True:
        print("\n1. Add Edge\n2. Perform DFS\n3. Exit")
        choice = int(input("Choose an option: "))
        if choice == 1:
            u = input("Enter starting node: ")
            v = input("Enter ending node: ")
            g.add_edge(u, v)
        elif choice == 2:
            start = input("Enter starting node for DFS: ")
            print("DFS Traversal: ", end='')
            g.dfs(start)
        elif choice == 3:
            break
        else:
            print("Invalid choice!")

if __name__ == "__main__":
    menu()
