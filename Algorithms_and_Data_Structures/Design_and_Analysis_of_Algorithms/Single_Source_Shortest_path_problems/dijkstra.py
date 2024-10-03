import sys

class Graph():
    def __init__(self, vertices):
        self.V = vertices
        self.graph = [[0 for _ in range(vertices)] for _ in range(vertices)]

    def printSolution(self, dist):
        print("Vertex \tDistance from Source")
        for node in range(self.V):
            print(f"{node} \t {dist[node]}")

    def minDistance(self, dist, sptSet):
        min_value = sys.maxsize
        min_index = 0
        for u in range(self.V):
            if dist[u] < min_value and not sptSet[u]:
                min_value = dist[u]
                min_index = u
        return min_index

    def dijkstra(self, src):
        dist = [sys.maxsize] * self.V
        dist[src] = 0
        sptSet = [False] * self.V

        for _ in range(self.V):
            x = self.minDistance(dist, sptSet)
            sptSet[x] = True

            for y in range(self.V):
                if self.graph[x][y] > 0 and not sptSet[y] and dist[y] > dist[x] + self.graph[x][y]:
                    dist[y] = dist[x] + self.graph[x][y]

        self.printSolution(dist)

if __name__ == "__main__":
    vertices = int(input("Enter the number of vertices: "))
    g = Graph(vertices)
    print("Enter the adjacency matrix (use 0 for no direct path):")
    for i in range(vertices):
        row = list(map(int, input().split()))
        g.graph[i] = row

    source = int(input("Enter the source vertex: "))
    g.dijkstra(source)
