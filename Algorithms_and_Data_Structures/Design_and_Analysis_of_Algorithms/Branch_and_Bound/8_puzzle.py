import numpy as np
from queue import PriorityQueue

class Node:
    def __init__(self, state, path_cost, level, parent=None):
        self.state = state  # Current state of the puzzle
        self.path_cost = path_cost  # Cost to reach this node
        self.level = level  # Depth of the node in the search tree
        self.parent = parent  # Parent node for path reconstruction

    def __lt__(self, other):
        # Comparison method for priority queue
        return (self.path_cost + self.level) < (other.path_cost + other.level)

def get_blank_position(state):
    # Returns the position of the blank (0) in the puzzle
    return np.argwhere(state == 0)[0]

def is_goal(state):
    # Checks if the current state is the goal state
    goal_state = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]])
    return np.array_equal(state, goal_state)

def get_successors(state):
    # Generates all possible successor states by moving the blank tile
    successors = []
    blank_pos = get_blank_position(state)
    x, y = blank_pos
    # Define possible moves (up, down, left, right)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    for dx, dy in directions:
        new_x, new_y = x + dx, y + dy
        # Check if the new position is within bounds
        if 0 <= new_x < 3 and 0 <= new_y < 3:
            new_state = state.copy()  # Create a copy of the current state
            # Swap the blank with the adjacent tile
            new_state[x, y], new_state[new_x, new_y] = new_state[new_x, new_y], new_state[x, y]
            successors.append(new_state)  # Add new state to successors
    return successors

def manhattan_distance(state):
    # Calculates the Manhattan distance heuristic for the given state
    distance = 0
    for i in range(3):
        for j in range(3):
            value = state[i, j]
            if value != 0:  # Ignore the blank tile
                # Calculate goal position for the current tile
                goal_x, goal_y = divmod(value - 1, 3)
                distance += abs(i - goal_x) + abs(j - goal_y)  # Add distances
    return distance

def branch_and_bound(initial_state):
    # Main function to perform the Branch and Bound search
    if is_goal(initial_state):
        return []  # If the initial state is the goal, return an empty path
    pq = PriorityQueue()  # Create a priority queue
    pq.put(Node(initial_state, 0, 0))  # Enqueue the initial state
    
    while not pq.empty():
        current_node = pq.get()  # Get the node with the lowest cost
        if is_goal(current_node.state):
            path = []
            while current_node:  # Backtrack to get the path
                path.append(current_node.state)
                current_node = current_node.parent
            return path[::-1]  # Return the path in correct order
        
        # Generate successors for the current state
        for successor in get_successors(current_node.state):
            path_cost = current_node.path_cost + 1  # Increment path cost
            heuristic_cost = manhattan_distance(successor)  # Calculate heuristic cost
            pq.put(Node(successor, path_cost, current_node.level + 1, current_node))  # Enqueue new node
            
    return None  # Return None if no solution is found

if __name__ == "__main__":
    print("Enter the initial state of the puzzle as 9 space-separated integers (0 for blank):")
    input_state = list(map(int, input().split()))
    initial_state = np.array(input_state).reshape(3, 3)  # Reshape input into 3x3 matrix

    solution_path = branch_and_bound(initial_state)  # Find solution path
    if solution_path:
        for step in solution_path:  # Print each step in the solution
            print(step)
    else:
        print("No solution found.")  # Indicate if no solution exists
