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
    node0 = Node(0)
    node1 = Node(1)
    node2 = Node(2)

    node0.neighbors.append(node1)
    node0.neighbors.append(node2)
    node1.neighbors.append(node0)
    node1.neighbors.append(node2)
    node2.neighbors.append(node0)
    node2.neighbors.append(node1)

    solution = Solution()
    clonedGraph = solution.cloneGraph(node0)

    # Print the original graph
    print("Original graph:")
    printed = {}
    printGraph(node0, printed)

    # Print the cloned graph
    print("\nCloned graph:")
    printed.clear()  # Reset printed map for cloned graph
    printGraph(clonedGraph, printed)
