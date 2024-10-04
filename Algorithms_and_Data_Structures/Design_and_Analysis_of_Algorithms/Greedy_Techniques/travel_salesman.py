from typing import List
from collections import defaultdict

# Define a constant for maximum integer value
INT_MAX = 2147483647

def findMinRoute(tsp: List[List[int]]):
    # Initialize total cost, counter, and indices for the current city
    total_cost = 0
    counter = 0
    j = 0  # Index for the next city to consider
    i = 0  # Index for the current city
    min_cost = INT_MAX  # Initialize minimum cost to maximum integer
    visited = defaultdict(int)  # Track visited cities using a default dictionary
    
    visited[0] = 1  # Mark the starting city (city 0) as visited
    route = [0] * len(tsp)  # Initialize the route array to track the selected route

    # Main loop to find the minimum cost route
    while i < len(tsp) and j < len(tsp[i]):
        if counter >= len(tsp[i]) - 1:
            break  # Exit if all cities have been visited

        # Check if the city has not been visited and is not the current city
        if j != i and (visited[j] == 0):
            # Update minimum cost and corresponding city if a cheaper path is found
            if tsp[i][j] < min_cost:
                min_cost = tsp[i][j]  # Update minimum cost
                route[counter] = j + 1  # Record the city in the route

        j += 1  # Move to the next city

        # If we've reached the end of the cities list, update total cost
        if j == len(tsp[i]):
            total_cost += min_cost  # Add the minimum cost found to total cost
            min_cost = INT_MAX  # Reset minimum cost for the next iteration
            visited[route[counter] - 1] = 1  # Mark the selected city as visited
            j = 0  # Reset index for the next iteration
            i = route[counter] - 1  # Move to the next city in the route
            counter += 1  # Increment counter for route

    # Check for the last city to return to the starting city
    i = route[counter - 1] - 1

    # Loop to find the minimum cost for returning to the starting city
    for j in range(len(tsp)):
        if (i != j) and tsp[i][j] < min_cost:
            min_cost = tsp[i][j]  # Update minimum cost if a cheaper path is found
            route[counter] = j + 1  # Record the returning city in the route

    total_cost += min_cost  # Add the cost to return to the total cost
    print("Minimum Cost is:", total_cost)  # Print the total minimum cost

if __name__ == "__main__":
    n = int(input("Enter the number of cities: "))  # Input number of cities
    tsp = []  # Initialize the adjacency matrix for the cities

    # Input the adjacency matrix with distances
    print("Enter the adjacency matrix (use -1 for no direct path):")
    for _ in range(n):
        row = list(map(int, input().split()))  # Input each row of the matrix
        tsp.append(row)  # Add the row to the adjacency matrix

    findMinRoute(tsp)  # Call the function to find the minimum route
