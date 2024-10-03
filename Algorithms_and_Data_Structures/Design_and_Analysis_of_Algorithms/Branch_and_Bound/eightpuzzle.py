import numpy as np
from queue import PriorityQueue

class Node:
    def __init__(self, state, path_cost, level, parent=None):
        self.state = state
        self.path_cost = path_cost
        self.level = level
        self.parent = parent

    def __lt__(self, other):
        return (self.path_cost + self.level) < (other.path_cost + other.level)

def get_blank_position(state):
    return np.argwhere(state == 0)[0]

def is_goal(state):
    goal_state = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]])
    return np.array_equal(state, goal_state)

def get_successors(state):
    successors = []
    blank_pos = get_blank_position(state)
    x, y = blank_pos
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dx, dy in directions:
        new_x, new_y = x + dx, y + dy
        if 0 <= new_x < 3 and 0 <= new_y < 3:
            new_state = state.copy()
            new_state[x, y], new_state[new_x, new_y] = new_state[new_x, new_y], new_state[x, y]
            successors.append(new_state)
    return successors

def manhattan_distance(state):
    distance = 0
    for i in range(3):
        for j in range(3):
            value = state[i, j]
            if value != 0:
                goal_x, goal_y = divmod(value - 1, 3)
                distance += abs(i - goal_x) + abs(j - goal_y)
    return distance

def branch_and_bound(initial_state):
    if is_goal(initial_state):
        return []
    pq = PriorityQueue()
    pq.put(Node(initial_state, 0, 0))
    while not pq.empty():
        current_node = pq.get()
        if is_goal(current_node.state):
            path = []
            while current_node:
                path.append(current_node.state)
                current_node = current_node.parent
            return path[::-1]
        for successor in get_successors(current_node.state):
            path_cost = current_node.path_cost + 1
            heuristic_cost = manhattan_distance(successor)
            pq.put(Node(successor, path_cost, current_node.level + 1, current_node))
    return None

if __name__ == "__main__":
    print("Enter the initial state of the puzzle as 9 space-separated integers (0 for blank):")
    input_state = list(map(int, input().split()))
    initial_state = np.array(input_state).reshape(3, 3)

    solution_path = branch_and_bound(initial_state)
    if solution_path:
        for step in solution_path:
            print(step)
    else:
        print("No solution found.")
