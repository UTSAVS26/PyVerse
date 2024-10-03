def floydWarshall(graph):
    dist = list(map(lambda i: list(map(lambda j: j, i)), graph))

    for k in range(len(graph)):
        for i in range(len(graph)):
            for j in range(len(graph)):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

    printSolution(dist)

def printSolution(dist):
    print("Following matrix shows the shortest distances between every pair of vertices:")
    for i in range(len(dist)):
        for j in range(len(dist)):
            if dist[i][j] == INF:
                print("%7s" % ("INF"), end=" ")
            else:
                print("%7d\t" % (dist[i][j]), end=' ')
            if j == len(dist) - 1:
                print()

if __name__ == "__main__":
    V = int(input("Enter the number of vertices: "))
    INF = float('inf')
    
    graph = []
    print("Enter the graph as an adjacency matrix (use 'INF' for no connection):")
    for i in range(V):
        row = list(map(lambda x: float('inf') if x == 'INF' else int(x), input().split()))
        graph.append(row)

    floydWarshall(graph)
