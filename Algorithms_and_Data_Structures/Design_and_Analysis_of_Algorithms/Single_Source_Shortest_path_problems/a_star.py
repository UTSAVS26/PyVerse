from collections import defaultdict
import heapq

class AStar:
    def __init__(self):
        self.adjacency_list = defaultdict(list)  # Using defaultdict for adjacency list

    def a_star(self, start, goal):
        open_set = []  # Priority queue for nodes to explore
        heapq.heappush(open_set, (0, start))
        open_set_set = {start}  # Set for quick lookup
        came_from = {}  # Lazy initialization
        g_scores = {start: 0}  # Initialize g_score for start
        f_scores = {start: self.manhattan_distance(start, goal)}  # Initialize f_score for start

        while open_set:
            current = heapq.heappop(open_set)[1]
            open_set_set.remove(current)  # Remove from the open set

            if current == goal:
                return self.reconstruct_path(came_from, current)  # Path construction when goal is found

            for neighbor in self.adjacency_list[current]:
                tentative_g_score = g_scores[current] + self.distance(current, neighbor)

                if neighbor not in g_scores or tentative_g_score < g_scores[neighbor]:
                    came_from[neighbor] = current
                    g_scores[neighbor] = tentative_g_score
                    f_scores[neighbor] = tentative_g_score + self.manhattan_distance(neighbor, goal)

                    # Only add the neighbor to the open set if it's not already there
                    if neighbor not in open_set_set:
                        heapq.heappush(open_set, (f_scores[neighbor], neighbor))
                        open_set_set.add(neighbor)  # Add to the set for O(1) lookup

        return []  # Return an empty path if no path is found

    def reconstruct_path(self, came_from, current):
        """Construct the path from start to goal."""
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.append(current)
        return total_path[::-1]  # Return the path from start to goal

    def distance(self, node1, node2):
        """Define your distance calculation here (e.g., uniform cost)."""
        return 1  # For a grid, assume each move has a cost of 1

    def manhattan_distance(self, node1, node2):
        """Calculate the Manhattan distance between two nodes."""
        x1, y1 = node1
        x2, y2 = node2
        return abs(x1 - x2) + abs(y1 - y2)

# Example usage
if __name__ == "__main__":
    astar = AStar()
    # Example graph setup, you would need to populate the adjacency_list
    astar.adjacency_list = {
        (0, 0): [(0, 1), (1, 0)],
        (0, 1): [(0, 0), (0, 2)],
        (0, 2): [(0, 1), (1, 2)],
        (1, 0): [(0, 0), (1, 1)],
        (1, 1): [(1, 0), (1, 2)],
        (1, 2): [(1, 1), (0, 2)],
    }

    start = (0, 0)
    goal = (1, 2)
    path = astar.a_star(start, goal)
    print("Path found:", path)

