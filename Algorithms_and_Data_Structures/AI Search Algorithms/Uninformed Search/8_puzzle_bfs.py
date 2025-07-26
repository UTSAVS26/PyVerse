import copy

goal_state = [[1,2,3],[4,5,6],[7,8,0]]

def find_zero(state):
    for i in range(3):
        for j in range(3):
            value = state[i][j]
            if value == 0:
                return i, j
    return None, None

def successor(state):
    moves = []
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    x, y = find_zero(state)
    
    for dx, dy in dirs:
        nx, ny = x + dx, y + dy
        if 0 <= nx < 3 and 0 <= ny < 3:
            new_state = copy.deepcopy(state)
            new_state[x][y], new_state[nx][ny] = new_state[nx][ny], new_state[x][y]
            moves.append(new_state)
    return moves

def bfs(state):
    queue = [(state, [])]
    close_list = set()
    
    while queue:
        current, path = queue.pop(0)
        
        if current == goal_state:
            for step in path + [current]:
                for row in step:
                    print(row)
                print()
            return
    
        if str(current) not in close_list:
            close_list.add(str(current))
            
            for child in successor(current):
                if str(child) not in close_list:
                    queue.append((child, path + [current]))
                    
    print("not found")
    return None
                    
def main(): 
    start_state = [[8, 1, 3],
                   [4, 0, 2],
                   [7, 6, 5]]

    bfs(start_state)

main()