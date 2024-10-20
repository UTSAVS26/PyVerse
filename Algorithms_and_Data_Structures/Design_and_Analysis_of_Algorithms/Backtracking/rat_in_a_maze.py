def print_solution(solution):
    """
    Print the solution matrix.
    """
    for row in solution:
        print(row)

def is_safe(maze, x, y):
    """
    Check if the given position (x, y) is safe to move to.
    Returns True if the position is within the maze and is not blocked.
    """
    n = len(maze)
    return 0 <= x < n and 0 <= y < n and maze[x][y] == 1

def solve_maze(maze):
    """
    Main function to solve the maze using backtracking.
    Returns the solution matrix if a path is found, otherwise returns None.
    """
    n = len(maze)
    # Create a solution matrix initialized with zeros
    solution = [[0 for _ in range(n)] for _ in range(n)]
    
    if solve_maze_util(maze, 0, 0, solution):
        return solution
    else:
        print("No solution exists.")
        return None

def solve_maze_util(maze, x, y, solution):
    """
    Recursive utility function to solve the maze.
    """
    n = len(maze)
    
    # Base case: if we've reached the destination
    if x == n - 1 and y == n - 1 and maze[x][y] == 1:
        solution[x][y] = 1
        return True
    
    # Check if the current position is safe to move
    if is_safe(maze, x, y):
        # Mark the current cell as part of the solution path
        solution[x][y] = 1
        
        # Move forward in x direction
        if solve_maze_util(maze, x + 1, y, solution):
            return True
        
        # If moving in x direction doesn't work, move down in y direction
        if solve_maze_util(maze, x, y + 1, solution):
            return True
        
        # If none of the above movements work, backtrack
        solution[x][y] = 0
        return False
    
    return False

# Example usage
if __name__ == "__main__":
    # 1 represents open path, 0 represents blocked path
    maze = [
        [1, 0, 0, 0],
        [1, 1, 0, 1],
        [0, 1, 0, 0],
        [1, 1, 1, 1]
    ]
    
    solution = solve_maze(maze)
    
    if solution:
        print("Solution found:")
        print_solution(solution)