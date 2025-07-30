import heapq

def a_star_search(adj, start, goal, heuristic):
    open_list = []
    heapq.heappush(open_list, (heuristic[start], 0, [start]))  # (f, g, path)
    
    cost_so_far = {start: 0}
    close_list = set()

    while open_list:
        _, g_n, path = heapq.heappop(open_list)
        u = path[-1]

        if u == goal:
            return path

        if u not in close_list:        
            close_list.add(u)

            for v, cost in adj[u]:
                g_new = g_n + cost
                if v not in cost_so_far or g_new < cost_so_far[v]:
                    cost_so_far[v] = g_new
                    f_new = g_new + heuristic[v]
                    heapq.heappush(open_list, (f_new, g_new, path + [v]))

    return None


def main():
    graph = {
        'A': [('B', 1), ('C', 3)],
        'B': [('D', 1), ('E', 4)],
        'C': [('F', 2)],
        'D': [],
        'E': [('F', 1)],
        'F': []
    }

    heuristic = {
        'A': 5,
        'B': 4,
        'C': 2,
        'D': 7,
        'E': 1,
        'F': 0
    }

    start, goal = 'A', 'F'
    path = a_star_search(graph, start, goal, heuristic)
    print("A* Path found:", path)

if __name__ == "__main__":
    main()

        