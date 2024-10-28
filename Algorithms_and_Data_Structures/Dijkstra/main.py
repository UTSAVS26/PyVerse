import networkx as nx
import matplotlib.pyplot as plt

# function to create graph
def graph_create():
    G = nx.Graph()
    nodes_num = int(input("Enter the number of nodes: "))
    for i in range(nodes_num):
        nodes_name = input(f"Enter the name of node {i+1} : ")
        
    edges = int(input("Enter the number of edges: "))
    for i in range(edges):
        edge_name = input(f"Enter edge {i+1} : (format: source, destination, value): ")
        source, destination, weight = edge_name.split()
        G.add_edge(source, destination, weight = int(weight))
        
    return G

# dijkstra algorithm implementation
def dijkstra(graph, start):
    distances = {node: float("inf") for node in graph.nodes} #dictionary to store shortest distances
    distances[start] = 0
    paths = {node: [] for node in graph.nodes} #dictionary to store shortest paths
    visited = set()

    while len(visited) < len(graph.nodes): #loop till all nodes are visited
        not_visited = {node: distances[node] for node in graph.nodes if node not in visited} #dictionary contains distances of unvisited nodes
        min_node = min(not_visited, key=not_visited.get) # to get node with minimum distance from start node
        visited.add(min_node)

        for neighbor, weight in graph[min_node].items():
            # If the distance to the neighbor through the current node is less than the previously known shortest distance to the neighbor
            if distances[min_node] + weight["weight"] < distances[neighbor]:
                # Update the shortest distance and path to the neighbor
                distances[neighbor] = distances[min_node] + weight["weight"]
                paths[neighbor] = paths[min_node] + [min_node]
                
    # After visiting all nodes, finalize the shortest paths by adding the destination node to each path

    paths = {node: path + [node] for node, path in paths.items() if path}

    return distances, paths

def visualise_dijkstra(graph, start):
    if start not in graph.nodes:
        print("Start node not found in graph")
        return
    
    distances, paths = dijkstra(graph, start)
    pos = nx.spring_layout(graph)
    plt.get_current_fig_manager().window.title("Dijkstra Algorithm Visualiser")
    nx.draw(graph, pos, with_labels = True, node_color = "lightblue", edgecolors="black", node_size = 500, font_size = 15, font_weight = "bold")
    labels = nx.get_edge_attributes(graph, "weight")
    nx.draw_networkx_edge_labels(graph, pos, edge_labels = labels, font_size = 8)

    plt.title("Dijkstra's Algorithm Visualisation")   
    print("Shortest distances from the start node:")
    for node, distance in distances.items():
        print(f"{node}: {distance}")

    print("Shortest paths from the start node:")
    for node, path in paths.items():
        print(f"{node}: {' -> '.join(path)}")
        
    plt.show()

if __name__ == "__main__":
    user_graph = graph_create()
    start_node = input("Enter the start node: ")
    visualise_dijkstra(user_graph, start_node)
    
        
        

    