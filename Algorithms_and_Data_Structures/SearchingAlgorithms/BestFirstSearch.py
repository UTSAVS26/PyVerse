import heapq

class Node:
    def __init__(self, name, cost=0):
        self.name = name
        self.cost = cost

    def __lt__(self, other):
        return self.cost < other.cost

def best_first_search(start, goal, graph):
    visited = set()
    priority_queue = []
    
    heapq.heappush(priority_queue, Node(start))
    
    while priority_queue:
        current_node = heapq.heappop(priority_queue)
        
        if current_node.name == goal:
            return f"Reached goal: {current_node.name}"

        if current_node.name in visited:
            continue
        
        visited.add(current_node.name)
        
        for neighbor, cost in graph.get(current_node.name, {}).items():
            if neighbor not in visited:
                heapq.heappush(priority_queue, Node(neighbor, cost))

    return "Goal not reachable"

# Example graph
graph = {
    'A': {'B': 1, 'C': 4},
    'B': {'A': 1, 'D': 2, 'E': 5},
    'C': {'A': 4, 'F': 3},
    'D': {'B': 2, 'F': 3},
    'E': {'B': 5},
    'F': {'C': 3, 'D': 3}
}

print(best_first_search('A', 'F', graph))
