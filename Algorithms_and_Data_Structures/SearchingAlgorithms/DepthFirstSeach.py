def depth_first_search(graph, start, visited=None):
    if visited is None:
        visited = set()
        
    visited.add(start)
    print(start)
    
    for neighbor in graph.get(start, []):
        if neighbor not in visited:
            depth_first_search(graph, neighbor, visited)

# Example graph
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['F'],
    'F': []
}

depth_first_search(graph, 'A')
