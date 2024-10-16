import unittest
from main import dijkstra

class DijkstraTestCase(unittest.TestCase):
    def test_shortest_path(self):
        graph = {
            'A': {'B': {'weight': 5}, 'C': {'weight': 3}},
            'B': {'A': {'weight': 5}, 'C': {'weight': 2}, 'D': {'weight': 1}},
            'C': {'A': {'weight': 3}, 'B': {'weight': 2}, 'D': {'weight': 4}, 'E': {'weight': 6}},
            'D': {'B': {'weight': 1}, 'C': {'weight': 4}, 'E': {'weight': 2}},
            'E': {'C': {'weight': 6}, 'D': {'weight': 2}}
        }
        start_node = 'A'
        expected_distances = {'A': 0, 'B': 3, 'C': 3, 'D': 4, 'E': 6}
        expected_paths = {'A': ['A'], 'B': ['A', 'B'], 'C': ['A', 'C'], 'D': ['A', 'B', 'D'], 'E': ['A', 'B', 'D', 'E']}
        
        distances, paths = dijkstra(graph, start_node)
        
        self.assertEqual(distances, expected_distances)
        self.assertEqual(paths, expected_paths)

    def test_disconnected_graph(self):
        graph = {
            'A': {'B': {'weight': 5}, 'C': {'weight': 3}},
            'B': {'A': {'weight': 5}, 'C': {'weight': 2}, 'D': {'weight': 1}},
            'C': {'A': {'weight': 3}, 'B': {'weight': 2}, 'D': {'weight': 4}, 'E': {'weight': 6}},
            'D': {'B': {'weight': 1}, 'C': {'weight': 4}, 'E': {'weight': 2}},
            'E': {'C': {'weight': 6}, 'D': {'weight': 2}},
            'F': {}  # Disconnected node
        }
        start_node = 'A'
        expected_distances = {'A': 0, 'B': 3, 'C': 3, 'D': 4, 'E': 6}
        expected_paths = {'A': ['A'], 'B': ['A', 'B'], 'C': ['A', 'C'], 'D': ['A', 'B', 'D'], 'E': ['A', 'B', 'D', 'E']}
        
        distances, paths = dijkstra(graph, start_node)
        
        self.assertEqual(distances, expected_distances)
        self.assertEqual(paths, expected_paths)

if __name__ == '__main__':
    unittest.main()