from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

# Create a sparse matrix representing the graph
graph = csr_matrix([[0, 2, 0, 6, 0],
                    [2, 0, 3, 8, 5],
                    [0, 3, 0, 0, 7],
                    [6, 8, 0, 0, 9],
                    [0, 5, 7, 9, 0]])

# Compute the MST
mst = minimum_spanning_tree(graph)

# Print the MST
print(mst.toarray().astype(int))