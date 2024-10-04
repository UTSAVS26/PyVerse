# Directions: D (down), L (left), R (right), U (up)
direction = "DLRU"
# Direction vectors for moving in the maze (down, left, right, up)
dr = [1, 0, 0, -1]
dc = [0, -1, 1, 0]

def is_valid(row, col, n, maze):
    # Check if the position (row, col) is within bounds and is a valid path (1)
    return 0 <= row < n and 0 <= col < n and maze[row][col] == 1

def find_path(row, col, maze, n, ans, current_path):
    # If the bottom-right corner of the maze is reached, append the current path to the answer list
    if row == n - 1 and col == n - 1:
        ans.append(current_path)
        return
    
    # Mark the current cell as visited
    maze[row][col] = 0

    # Explore all possible directions (down, left, right, up)
    for i in range(4):
        next_row = row + dr[i]  # Calculate the new row index
        next_col = col + dc[i]  # Calculate the new column index

        # If the next position is valid, continue the search
        if is_valid(next_row, next_col, n, maze):
            find_path(next_row, next_col, maze, n, ans, current_path + direction[i])  # Append direction to the path
    
    # Backtrack: Unmark the current cell
    maze[row][col] = 1

def main():
    n = int(input("Enter the size of the maze (n x n): "))  # Get the size of the maze from the user
    print("Enter the maze row by row (1 for path, 0 for block):")
    # Read the maze input as a list of lists
    maze = [list(map(int, input().split())) for _ in range(n)]

    result = []  # List to store all valid paths
    current_path = ""  # String to store the current path

    # Check if the starting and ending points are valid paths
    if maze[0][0] != 0 and maze[n - 1][n - 1] != 0:
        find_path(0, 0, maze, n, result, current_path)  # Start finding paths from (0, 0)

    # Print the result: either valid paths or -1 if no paths found
    if not result:
        print(-1)  # No valid paths found
    else:
        print("Valid paths:", " ".join(result))  # Print all valid paths

if __name__ == "__main__":
    main()  # Run the main function
