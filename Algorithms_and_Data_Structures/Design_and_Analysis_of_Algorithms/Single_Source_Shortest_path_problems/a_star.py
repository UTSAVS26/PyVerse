import heapq

class Graph:
    def __init__(self):
        self.adjacency_list = {}

    def add_edge(self, u, v, cost):
        if u not in self.adjacency_list:
            self.adjacency_list[u] = []
        if v not in self.adjacency_list:
            self.adjacency_list[v] = []
        self.adjacency_list[u].append((v, cost))
        self.adjacency_list[v].append((u, cost))  # For undirected graph

def heuristic(a, b):
    # Example heuristic function (can be replaced with a real heuristic)
    # Here we use a simple Manhattan distance as an example for 2D coordinates
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star_search(graph, start, goal):
    open_set = []
    heapq.heappush(open_set, (0, start))  # (f_score, node)
    
    came_from = {}
    g_score = {node: float('inf') for node in graph.adjacency_list}
    g_score[start] = 0
    
    f_score = {node: float('inf') for node in graph.adjacency_list}
    f_score[start] = heuristic(start, goal)

    while open_set:
        current = heapq.heappop(open_set)[1]

        if current == goal:
            # Reconstruct the path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]  # Return reversed path

        for neighbor, cost in graph.adjacency_list.get(current, []):
            tentative_g_score = g_score[current] + cost
            
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                
                if all(neighbor != n[1] for n in open_set):
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None  # No path found

# Example usage:
if __name__ == "__main__":
    g = Graph()
    g.add_edge((0, 0), (0, 1), 1)
    g.add_edge((0, 0), (1, 0), 1)
    g.add_edge((0, 1), (1, 1), 1)
    g.add_edge((1, 0), (1, 1), 1)
    g.add_edge((1, 1), (1, 2), 1)
    g.add_edge((1, 0), (2, 0), 1)
    g.add_edge((1, 2), (2, 2), 1)
    g.add_edge((2, 0), (2, 1), 1)
    g.add_edge((2, 1), (2, 2), 1)

    start = (0, 0)
    goal = (2, 2)
    path = a_star_search(g, start, goal)

    print("Path found:", path)

