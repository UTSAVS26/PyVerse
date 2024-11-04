class Reversal:
    def function(self, graph):
        # Initialize a list of empty lists to store the reversed graph
        sol = [[] for _ in range(len(graph))]
        
        # Iterate over each node and its edges
        for i in range(len(graph)):
            for x in graph[i]:
                # Add an edge in the reverse direction
                sol[x].append(i)
        
        return sol

if __name__ == "__main__":
    # Example graph as an adjacency list
    graph = [[1, 2], [4], [4], [1, 2], [3]]
    ob = Reversal()
    result = ob.function(graph)
    # Print the reversed graph 
    for vec in result:
        print("[", " ".join(map(str, vec)), "]")
