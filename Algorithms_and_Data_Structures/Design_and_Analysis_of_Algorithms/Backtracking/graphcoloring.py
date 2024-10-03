V = 4

def print_solution(color):
    print("Solution Exists: Following are the assigned colors")
    print(" ".join(map(str, color)))

def is_safe(v, graph, color, c):
    for i in range(V):
        if graph[v][i] and c == color[i]:
            return False
    return True

def graph_coloring_util(graph, m, color, v):
    if v == V:
        return True

    for c in range(1, m + 1):
        if is_safe(v, graph, color, c):
            color[v] = c

            if graph_coloring_util(graph, m, color, v + 1):
                return True

            color[v] = 0

    return False

def graph_coloring(graph, m):
    color = [0] * V

    if not graph_coloring_util(graph, m, color, 0):
        print("Solution does not exist")
        return False

    print_solution(color)
    return True

def main():
    print("Enter the number of vertices:")
    global V
    V = int(input())
    
    graph = []
    print("Enter the adjacency matrix (0 for no edge, 1 for edge):")
    for _ in range(V):
        row = list(map(int, input().split()))
        graph.append(row)

    m = int(input("Enter the number of colors: "))
    
    graph_coloring(graph, m)

if __name__ == "__main__":
    main()
