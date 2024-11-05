"""
Problem: You are tasked with painting houses. Each house can be painted in one of k colors,
and no two adjacent houses can have the same color. Find the minimum cost to paint all houses.
"""
def paint_house(costs):
    if not costs:
        return 0

    n = len(costs)
    k = len(costs[0])
    dp = costs[0][:]

    for i in range(1, n):
        prev_dp = dp[:]
        for j in range(k):
            dp[j] = costs[i][j] + min(prev_dp[m] for m in range(k) if m != j)

    return min(dp)

# Example usage
costs = [[17, 2, 17], [16, 16, 5], [14, 3, 19]]
print(f"Minimum cost to paint all houses: {paint_house(costs)}")  # Output: Minimum cost to paint all houses: 10
