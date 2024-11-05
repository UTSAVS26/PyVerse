from collections import deque

def is_bipartite(adj_matrix, V):
    # Initialize colors array with -1 (uncolored)
    color = [-1] * V
    # Process each component of the graph
    for start in range(V):
        # If the vertex is already colored, skip it
        if color[start] != -1:
            continue
        # Start BFS from this node
        q = deque([start])
        color[start] = 0  # Color the starting vertex with 0
        while q:
            node = q.popleft()
        
            for neighbor in range(V):
                # Check if there's an edge between node and neighbor
                if adj_matrix[node][neighbor] == 1:
                    # If the neighbor hasn't been colored, color it with the opposite color
                    if color[neighbor] == -1:
                        color[neighbor] = 1 - color[node]
                        q.append(neighbor)
                    # If the neighbor has the same color as the current node, the graph is not bipartite
                    elif color[neighbor] == color[node]:
                        return False
    # If we successfully colored the graph, it's bipartite
    return True

def main():
    V = int(input("Enter the number of vertices: "))
    adj_matrix = [[0] * V for _ in range(V)]
    print("Enter the adjacency matrix:")
    for i in range(V):
        adj_matrix[i] = list(map(int, input().split()))
    
    if is_bipartite(adj_matrix, V):
        print("The graph is bipartite.")
    else:
        print("The graph is not bipartite.")

if __name__ == "__main__":
    main()
