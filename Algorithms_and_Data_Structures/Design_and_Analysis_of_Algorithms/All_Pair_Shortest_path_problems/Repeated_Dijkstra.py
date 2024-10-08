#input: graph represented as adjacency list, source vertex
#output: shortest path from source vertex to all other vertices
import heapq
import math

def all_pair_shortest_path(graph, source):
    #initialize distance from source to all other vertices
    distance = {}
    for vertex in graph:
        distance[vertex] = math.inf
    distance[source] = 0

    #initialize heap
    heap = []
    heapq.heappush(heap, (0, source))

    #initialize parent
    parent = {}
    parent[source] = None

    #initialize visited set
    visited = set()

    #initialize shortest path
    shortest_path = {}
    shortest_path[source] = [source]

    while len(heap) > 0:
        #extract min distance vertex
        min_distance, min_vertex = heapq.heappop(heap)

        #relax all edges from min_vertex
        for neighbor, weight in graph[min_vertex]:
            if distance[min_vertex] + weight < distance[neighbor]:
                distance[neighbor] = distance[min_vertex] + weight
                heapq.heappush(heap, (distance[neighbor], neighbor))
                parent[neighbor] = min_vertex
                shortest_path[neighbor] = shortest_path[min_vertex] + [neighbor]

    return distance, parent, shortest_path

#driver code
if __name__ == '__main__':
    graph = {
        'A': [('B', 3), ('C', 6), ('D', 1)],
        'B': [('A', 3), ('C', 2), ('D', 1)],
        'C': [('A', 6), ('B', 2), ('D', 2)],
        'D': [('A', 1), ('B', 1), ('C', 2)]
    }
    for vertex in graph:
        source = vertex
        distance, parent, shortest_path = all_pair_shortest_path(graph, source)
        print(distance)
        print(shortest_path)
        
