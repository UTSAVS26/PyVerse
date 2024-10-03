from typing import List
from collections import defaultdict

INT_MAX = 2147483647

def findMinRoute(tsp: List[List[int]]):
    total_cost = 0
    counter = 0
    j = 0
    i = 0
    min_cost = INT_MAX
    visited = defaultdict(int)
    
    visited[0] = 1
    route = [0] * len(tsp)

    while i < len(tsp) and j < len(tsp[i]):
        if counter >= len(tsp[i]) - 1:
            break

        if j != i and (visited[j] == 0):
            if tsp[i][j] < min_cost:
                min_cost = tsp[i][j]
                route[counter] = j + 1

        j += 1

        if j == len(tsp[i]):
            total_cost += min_cost
            min_cost = INT_MAX
            visited[route[counter] - 1] = 1
            j = 0
            i = route[counter] - 1
            counter += 1

    i = route[counter - 1] - 1

    for j in range(len(tsp)):
        if (i != j) and tsp[i][j] < min_cost:
            min_cost = tsp[i][j]
            route[counter] = j + 1

    total_cost += min_cost
    print("Minimum Cost is:", total_cost)

if __name__ == "__main__":
    n = int(input("Enter the number of cities: "))
    tsp = []

    print("Enter the adjacency matrix (use -1 for no direct path):")
    for _ in range(n):
        row = list(map(int, input().split()))
        tsp.append(row)

    findMinRoute(tsp)
