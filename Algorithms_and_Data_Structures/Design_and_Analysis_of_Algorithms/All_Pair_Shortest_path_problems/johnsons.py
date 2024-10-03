from collections import defaultdict

INT_MAX = float('Inf')

def Min_Distance(dist, visit):
    minimum, minVertex = INT_MAX, -1
    for vertex in range(len(dist)):
        if minimum > dist[vertex] and not visit[vertex]:
            minimum, minVertex = dist[vertex], vertex
    return minVertex

def Dijkstra_Algorithm(graph, Altered_Graph, source):
    tot_vertices = len(graph)
    sptSet = defaultdict(lambda: False)
    dist = [INT_MAX] * tot_vertices
    dist[source] = 0

    for _ in range(tot_vertices):
        curVertex = Min_Distance(dist, sptSet)
        sptSet[curVertex] = True

        for vertex in range(tot_vertices):
            if (not sptSet[vertex] and 
                dist[vertex] > dist[curVertex] + Altered_Graph[curVertex][vertex] and 
                graph[curVertex][vertex] != 0):
                dist[vertex] = dist[curVertex] + Altered_Graph[curVertex][vertex]

    for vertex in range(tot_vertices):
        print(f'Vertex {vertex}: {dist[vertex]}')

def BellmanFord_Algorithm(edges, graph, tot_vertices):
    dist = [INT_MAX] * (tot_vertices + 1)
    dist[tot_vertices] = 0

    for i in range(tot_vertices):
        edges.append([tot_vertices, i, 0])

    for _ in range(tot_vertices):
        for (source, destn, weight) in edges:
            if dist[source] != INT_MAX and dist[source] + weight < dist[destn]:
                dist[destn] = dist[source] + weight

    return dist[0:tot_vertices]

def JohnsonAlgorithm(graph):
    edges = []
    for i in range(len(graph)):
        for j in range(len(graph[i])):
            if graph[i][j] != 0:
                edges.append([i, j, graph[i][j]])

    Alter_weights = BellmanFord_Algorithm(edges, graph, len(graph))
    Altered_Graph = [[0 for _ in range(len(graph))] for _ in range(len(graph))]

    for i in range(len(graph)):
        for j in range(len(graph[i])):
            if graph[i][j] != 0:
                Altered_Graph[i][j] = graph[i][j] + Alter_weights[i] - Alter_weights[j]

    print('Modified Graph:', Altered_Graph)

    for source in range(len(graph)):
        print(f'\nShortest Distance with vertex {source} as the source:\n')
        Dijkstra_Algorithm(graph, Altered_Graph, source)

if __name__ == "__main__":
    V = int(input("Enter the number of vertices: "))
    graph = []
    print("Enter the graph as an adjacency matrix (use 0 for no connection):")
    for _ in range(V):
        row = list(map(int, input().split()))
        graph.append(row)

    JohnsonAlgorithm(graph)
