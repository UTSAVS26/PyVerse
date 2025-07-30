import copy

goal_state = [[1,2,3],[4,5,6],[7,8,0]]

def find_zero(state):
    for i in range(3):
        for j in range(3):
            if state[i][j] == 0:
                return i, j
    return None, None

def successor(state):
    moves = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    x, y = find_zero(state)
    
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        
        if 0 <= nx < 3 and 0 <= ny < 3:
            new_state = copy.deepcopy(state)
            new_state[x][y], new_state[nx][ny] = new_state[nx][ny], new_state[x][y]
            
            moves.append(new_state)
    return moves

def dfs(state, max_depth = 30):

    stack = []
    stack.append((state, [])) # node, path
    
    close_list = set()

    while stack:
        current, path = stack.pop()
        
        if current == goal_state:
            for step in path + [current]:
                for row in step:
                    print(row)
                print()
            return
        
        if len(path) > max_depth:
            continue
        
        close_list.add(str(current))
        
        for child in successor(current):
            if str(child) not in close_list:
                stack.append((child, path + [current]))

    print("No path Found!")
    return None

def main(): 
    start_state = [[8, 1, 3],
                   [4, 0, 2],
                   [7, 6, 5]]

    dfs(start_state)

main()