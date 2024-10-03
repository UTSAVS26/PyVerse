class Graph(): 
    def __init__(self, vertices): 
        self.graph = [[0 for _ in range(vertices)] for _ in range(vertices)] 
        self.V = vertices 

    def is_safe(self, v, pos, path): 
        if self.graph[path[pos-1]][v] == 0: 
            return False
        return v not in path[:pos]

    def ham_cycle_util(self, path, pos): 
        if pos == self.V: 
            return self.graph[path[pos-1]][path[0]] == 1
        for v in range(1, self.V): 
            if self.is_safe(v, pos, path): 
                path[pos] = v 
                if self.ham_cycle_util(path, pos + 1): 
                    return True
                path[pos] = -1
        return False

    def ham_cycle(self): 
        path = [-1] * self.V 
        path[0] = 0
        if not self.ham_cycle_util(path, 1): 
            print("Solution does not exist\n")
            return False
        self.print_solution(path) 
        return True

    def print_solution(self, path): 
        print("Solution Exists: Following is one Hamiltonian Cycle")
        print(" -> ".join(map(str, path + [path[0]])))  # Include start point to complete the cycle

def main():
    vertices = int(input("Enter the number of vertices: "))
    g = Graph(vertices)
    print("Enter the adjacency matrix (0 for no edge, 1 for edge):")
    for i in range(vertices):
        row = list(map(int, input().split()))
        g.graph[i] = row

    g.ham_cycle()

if __name__ == "__main__":
    main()
