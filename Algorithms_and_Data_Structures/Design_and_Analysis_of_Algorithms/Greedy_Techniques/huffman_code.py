import heapq  # Import the heapq module to use a priority queue

class Node: 
    def __init__(self, freq, symbol, left=None, right=None): 
        self.freq = freq  # Frequency of the symbol
        self.symbol = symbol  # Character symbol
        self.left = left  # Left child
        self.right = right  # Right child
        self.huff = ''  # Huffman code for the symbol

    def __lt__(self, nxt): 
        return self.freq < nxt.freq  # Comparison based on frequency for priority queue

def printNodes(node, val=''): 
    newVal = val + str(node.huff)  # Append the Huffman code
    if node.left: 
        printNodes(node.left, newVal)  # Traverse left child
    if node.right: 
        printNodes(node.right, newVal)  # Traverse right child
    if not node.left and not node.right: 
        print(f"{node.symbol} -> {newVal}")  # Print the symbol and its corresponding Huffman code

if __name__ == "__main__":
    # Input characters and their corresponding frequencies
    chars = input("Enter characters separated by spaces: ").split()
    freq = list(map(int, input("Enter corresponding frequencies separated by spaces: ").split()))

    nodes = [] 
    # Create a priority queue with nodes for each character
    for x in range(len(chars)): 
        heapq.heappush(nodes, Node(freq[x], chars[x])) 

    # Build the Huffman tree
    while len(nodes) > 1: 
        left = heapq.heappop(nodes)  # Pop the two nodes with the smallest frequency
        right = heapq.heappop(nodes) 
        left.huff = 0  # Assign 0 to the left child
        right.huff = 1  # Assign 1 to the right child
        # Create a new node with combined frequency
        newNode = Node(left.freq + right.freq, left.symbol + right.symbol, left, right) 
        heapq.heappush(nodes, newNode)  # Push the new node back into the priority queue

    print("Huffman Codes:")  # Output the generated Huffman codes
    printNodes(nodes[0])  # Print the Huffman codes starting from the root node
