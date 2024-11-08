def coin_change_greedy(coins, amount):
    # Sort the coins in descending order for greedy approach
    coins.sort(reverse=True)
    
    # Initialize variables to track results
    coin_count = 0  # Total number of coins used
    remaining_amount = amount  # Amount left to be made
    coin_usage = {}  # Dictionary to store how many of each coin is used

    # Iterate through each coin denomination
    for coin in coins:
        # Calculate how many of the current coin can be used
        if coin <= remaining_amount:
            num_coins = remaining_amount // coin  # Integer division
            coin_count += num_coins  # Add to total coin count
            remaining_amount -= num_coins * coin  # Update remaining amount
            
            # Store the coin usage in the dictionary
            coin_usage[coin] = num_coins

        # If we've reached the target amount, exit the loop
        if remaining_amount == 0:
            break

    # Check if we were able to make the full amount
    if remaining_amount > 0:
        return -1, {}  # Return -1 and empty dict if not possible
    else:
        return coin_count, coin_usage  # Return total coins and usage dict

# Example usage
if __name__ == "__main__":
    coins = [25, 10, 5, 1]  # Available coin denominations
    amount = 67  # Target amount to make
    
    # Call the greedy coin change function
    result, coin_usage = coin_change_greedy(coins, amount)
    
    # Print the results
    if result == -1:
        print(f"It's not possible to make {amount} with the given coins.")
    else:
        print(f"Minimum number of coins needed to make {amount}: {result}")
        print("Coin usage:")
        for coin, count in coin_usage.items():
            print(f"{count} coin(s) of {coin}")