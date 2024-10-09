def coin_change_greedy(coins, amount):
    # Sort the coins in descending order
    coins.sort(reverse=True)
    
    # Initialize variables
    coin_count = 0
    remaining_amount = amount
    coin_usage = {}

    # Iterate through each coin denomination
    for coin in coins:
        # Calculate how many of the current coin can be used
        if coin <= remaining_amount:
            num_coins = remaining_amount // coin
            coin_count += num_coins
            remaining_amount -= num_coins * coin
            
            # Store the coin usage
            coin_usage[coin] = num_coins

        # If we've reached the target amount, break the loop
        if remaining_amount == 0:
            break

    # Check if we were able to make the full amount
    if remaining_amount > 0:
        return -1, {}
    else:
        return coin_count, coin_usage

# Example usage
if __name__ == "__main__":
    coins = [25, 10, 5, 1]  # Available coin denominations
    amount = 67  # Target amount
    
    result, coin_usage = coin_change_greedy(coins, amount)
    
    if result == -1:
        print(f"It's not possible to make {amount} with the given coins.")
    else:
        print(f"Minimum number of coins needed to make {amount}: {result}")
        print("Coin usage:")
        for coin, count in coin_usage.items():
            print(f"{count} coin(s) of {coin}")