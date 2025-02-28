def rob(houses):
    # If there are no houses, return 0
    if not houses:
        return 0
    # If there is only one house, return its value
    if len(houses) == 1:
        return houses[0]

    # Create a table to store the maximum money that can be robbed up to each house
    dp = [0] * len(houses)
    dp[0] = houses[0]  # Max money for the first house
    dp[1] = max(houses[0], houses[1])  # Max money for the first two houses

    # Fill the table in a bottom-up manner
    for i in range(2, len(houses)):
        # For each house, decide whether to rob it (dp[i-2] + houses[i]) or not (dp[i-1])
        dp[i] = max(dp[i - 1], dp[i - 2] + houses[i])

    return dp[-1]

# Test the function
houses = [2, 7, 9, 3, 1]
print(f"Maximum money that can be robbed: {rob(houses)}")
