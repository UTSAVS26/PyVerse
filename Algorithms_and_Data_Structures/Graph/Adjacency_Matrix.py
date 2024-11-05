def create_graph(adj, no_of_nodes):
    for i in range(no_of_nodes):
        val = int(input(f"\nEnter the number of neighbors of {i}: "))
        print(f"\nEnter the neighbors of {i} (0-based indices): ")
        for j in range(no_of_nodes):
            adj[i][j] = 0
        for j in range(val):
            neighbor = int(input())
            adj[i][neighbor] = 1

def display_graph(adj, no_of_nodes):
    print("\nThe adjacency matrix is:")
    print("\t", end="")
    for i in range(no_of_nodes):
        print(f"v{i + 1}\t", end="")
    print()
    for i in range(no_of_nodes):
        print(f"v{i + 1}\t", end="")
        for j in range(no_of_nodes):
            print(f"{adj[i][j]}\t", end="")
        print()

def main():
    adj = [[0] * 10 for _ in range(10)]
    no_of_nodes = int(input("\nEnter the number of nodes in G: "))
    create_graph(adj, no_of_nodes)
    print("\nThe graph is: ")
    display_graph(adj, no_of_nodes)

if __name__ == "__main__":
    main()
