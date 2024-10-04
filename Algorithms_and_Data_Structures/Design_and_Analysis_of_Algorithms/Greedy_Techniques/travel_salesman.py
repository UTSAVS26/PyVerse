from typing import List
from collections import defaultdict

INT_MAX = 2147483647  # Define a constant for the maximum integer value

def findMinRoute(tsp: List[List[int]]):
    total_cost = 0  # Initialize total cost to 0
    counter = 0  # Counter for the number of cities visited
    j = 0  # Column index for checking cities
    i = 0  # Row index for the current city
    min_cost = INT_MAX  # Initialize minimum cost to maximum value
    visited = defaultdict(int)  # Keep track of visited cities
    
    visited[0] = 1  # Start from the first city (city 0)
    route = [0] * len(tsp)  # Initialize route list to store the order of visited cities

    while i < len(tsp) and j < len(tsp[i]):
        if counter >= len(tsp[i]) - 1:  # Check if all cities have been visited
            break

        # If j is not the current city and not visited
        if j != i and (visited[j] == 0):
            # Update minimum cost if a cheaper route is found
            if tsp[i][j] < min_cost:
                min_cost = tsp[i][j]
                route[counter] = j + 1  # Store the city number in route (1-based index)

        j += 1  # Move to the next city

        if j == len(tsp[i]):  # If all cities in the current row have been checked
            total_cost += min_cost  # Add minimum cost to total cost
            min_cost = INT_MAX  # Reset minimum cost for the next city
            visited[route[counter] - 1] = 1  # Mark the city as visited
            j = 0  # Reset column index
            i = route[counter] - 1  # Move to the next city
            counter += 1  # Increment the visited city counter

    # Handle the return trip to the starting city
    i = route[counter - 1] - 1

    for j in range(len(tsp)):
        if (i != j) and tsp[i][j] < min_cost:
            min_cost = tsp[i][j]
            route[counter] = j + 1

    total_cost += min_cost  # Add the cost of returning to the start city
    print("Minimum Cost is:", total_cost)  # Print the total minimum cost

if __name__ == "__main__":
    n = int(input("Enter the number of cities: "))  # Input for number of cities
    tsp = []  # Initialize adjacency matrix for TSP

    print("Enter the adjacency matrix (use -1 for no direct path):")
    for _ in range(n):
        row = list(map(int, input().split()))  # Input for each row of the adjacency matrix
        tsp.append(row)

    findMinRoute(tsp)  # Call the function to find the minimum cost route
