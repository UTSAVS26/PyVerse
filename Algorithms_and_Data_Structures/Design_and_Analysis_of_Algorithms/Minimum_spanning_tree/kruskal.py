class Graph: 
    def __init__(self, vertices): 
        self.V = vertices 
        self.graph = [] 

    def add_edge(self, u, v, w): 
        self.graph.append([u, v, w]) 

    def find(self, parent, i): 
        if parent[i] != i:
            parent[i] = self.find(parent, parent[i]) 
        return parent[i] 

    def union(self, parent, rank, x, y): 
        if rank[x] < rank[y]: 
            parent[x] = y 
        elif rank[x] > rank[y]: 
            parent[y] = x 
        else: 
            parent[y] = x 
            rank[x] += 1

    def kruskal_mst(self): 
        result = [] 
        i, e = 0, 0 
        self.graph.sort(key=lambda item: item[2]) 
        parent = list(range(self.V)) 
        rank = [0] * self.V 

        while e < self.V - 1: 
            u, v, w = self.graph[i] 
            i += 1
            x = self.find(parent, u) 
            y = self.find(parent, v) 
            if x != y: 
                e += 1
                result.append([u, v, w]) 
                self.union(parent, rank, x, y) 

        minimum_cost = sum(weight for _, _, weight in result)
        print("Edges in the constructed MST") 
        for u, v, weight in result: 
            print(f"{u} -- {v} == {weight}") 
        print("Minimum Spanning Tree Cost:", minimum_cost) 

def main():
    vertices = int(input("Enter the number of vertices: "))
    g = Graph(vertices)
    edges = int(input("Enter the number of edges: "))
    print("Enter each edge in the format 'u v w' where u and v are vertices and w is the weight:")
    
    for _ in range(edges):
        u, v, w = map(int, input().split())
        g.add_edge(u, v, w)

    g.kruskal_mst()

if __name__ == '__main__': 
    main()
