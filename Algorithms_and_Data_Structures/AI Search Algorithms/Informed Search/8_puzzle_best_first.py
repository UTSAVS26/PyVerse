import heapq
import copy

goal_state = [[1, 2, 3],
              [4, 5, 6],
              [7, 8, 0]]

def heuristic(state):
    h = 0
    for i in range(3):
        for j in range(3):
            value = state[i][j]
            if value != 0:
                goal_x = (value - 1) // 3
                goal_y = (value - 1) % 3
                h += abs(i - goal_x) + abs(j - goal_y)
    return h

def find_zero(state):
    for i in range(3):
        for j in range(3):
            if state[i][j] == 0:
                return i, j

def successors(state):
    moves = []
    x, y = find_zero(state)
    directions = [(-1,0), (1,0), (0,-1), (0,1)]
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < 3 and 0 <= ny < 3:
            new_state = copy.deepcopy(state)
            new_state[x][y], new_state[nx][ny] = new_state[nx][ny], new_state[x][y]
            moves.append(new_state)
    return moves

def best_first_search(start):
    heap = [(heuristic(start), start, [])]
    visited = set()
    
    while heap:
        h, current, path = heapq.heappop(heap)
        
        if current == goal_state:
            i = 0
            print("Start State: ")
            for step in path + [current]:
                if i > 0:
                    print("Move:", i)
                for row in step:
                    print(row)
                print()
                i += 1
            return
        
        if str(current) not in visited:
            visited.add(str(current))
            
            for child in successors(current):
                if str(child) not in visited:
                    heapq.heappush(heap, (heuristic(child), child, path + [current]))

def main():
    start_state = [[8, 1, 3],
                   [4, 0, 2],
                   [7, 6, 5]]

    best_first_search(start_state)

if __name__ == "__main__":
    main()
