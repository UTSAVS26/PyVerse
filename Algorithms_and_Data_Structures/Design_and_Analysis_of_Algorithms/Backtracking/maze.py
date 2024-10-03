direction = "DLRU"
dr = [1, 0, 0, -1]
dc = [0, -1, 1, 0]

def is_valid(row, col, n, maze):
    return 0 <= row < n and 0 <= col < n and maze[row][col] == 1

def find_path(row, col, maze, n, ans, current_path):
    if row == n - 1 and col == n - 1:
        ans.append(current_path)
        return
    maze[row][col] = 0

    for i in range(4):
        next_row = row + dr[i]
        next_col = col + dc[i]

        if is_valid(next_row, next_col, n, maze):
            find_path(next_row, next_col, maze, n, ans, current_path + direction[i])
    
    maze[row][col] = 1

def main():
    n = int(input("Enter the size of the maze (n x n): "))
    print("Enter the maze row by row (1 for path, 0 for block):")
    maze = [list(map(int, input().split())) for _ in range(n)]

    result = []
    current_path = ""

    if maze[0][0] != 0 and maze[n - 1][n - 1] != 0:
        find_path(0, 0, maze, n, result, current_path)

    if not result:
        print(-1)
    else:
        print("Valid paths:", " ".join(result))

if __name__ == "__main__":
    main()
