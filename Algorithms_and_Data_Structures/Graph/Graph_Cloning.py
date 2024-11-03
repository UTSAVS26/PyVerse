class Node:
    def __init__(self, _val):
        self.val = _val
        self.neighbors = []

class Solution:
    def __init__(self):
        self.visited = {}

    def cloneGraph(self, node):
        if not node:
            return None
        if node in self.visited:
            return self.visited[node]
        cloneNode = Node(node.val)
        self.visited[node] = cloneNode
        for neighbor in node.neighbors:
            cloneNode.neighbors.append(self.cloneGraph(neighbor))
        return cloneNode

def printGraph(node, printed):
    if not node or printed.get(node, False):
        return
    printed[node] = True  # Mark this node as printed
    print(f"Node: {node.val} Neighbors: ", end="")
    for neighbor in node.neighbors:
        print(neighbor.val, end=" ")
    print()
    for neighbor in node.neighbors:
        printGraph(neighbor, printed)

if __name__ == "__main__":
    # Creating the graph with nodes 50, 60, and 70
    node50 = Node(50)
    node60 = Node(60)
    node70 = Node(70)

    # Establishing the connections (edges) between nodes
    node50.neighbors.append(node60)
    node50.neighbors.append(node70)
    node60.neighbors.append(node50)
    node60.neighbors.append(node70)
    node70.neighbors.append(node50)
    node70.neighbors.append(node60)

    solution = Solution()
    clonedGraph = solution.cloneGraph(node50)

    # Print the original graph
    print("Original graph:")
    printed = {}
    printGraph(node50, printed)

    # Print the cloned graph
    print("\nCloned graph:")
    printed.clear()  # Reset printed map for cloned graph
    printGraph(clonedGraph, printed)
