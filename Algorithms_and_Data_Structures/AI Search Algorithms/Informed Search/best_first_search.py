import heapq

def best_first_search(adj, start, goal, heuristic):
    
    pq = []
    close_list = set()
    
    heapq.heappush(pq, (heuristic[start], [start]))
    
    while len(pq) != 0:
        _, path = heapq.heappop(pq)
        
        u = path[-1]
        
        if u == goal:
            return path
        
        if u not in close_list:
            close_list.add(u)
            
            for v in adj[u]:
                if v not in close_list:
                    heapq.heappush(pq, (heuristic[v], path + [v]))
                    
    return None


def main():
    adj = {
        'A': ['B', 'C'],
        'B': ['D', 'E'],
        'C': ['F'],
        'D': [],
        'E': ['F'],
        'F': []
    }
    
    heuristic = {
        'A': 5,
        'B': 4,
        'C': 3,
        'D': 7,
        'E': 2,
        'F': 0  # Goal node has heuristic 0
    }

    start, goal = 'A', 'F'
    path = best_first_search(adj, start, goal, heuristic)
    print(path)

if __name__ == "__main__":
    main()
        