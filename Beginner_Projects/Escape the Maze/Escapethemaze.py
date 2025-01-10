import random

def generate_maze(size):
    """Generate a maze of given size with treasures, traps, and an exit."""
    maze = {}
    for row in range(size):
        for col in range(size):
            maze[(row, col)] = random.choice(["Empty", "Treasure", "Trap"])
    maze[(0, 0)] = "Start"
    maze[(size - 1, size - 1)] = "Exit"
    return maze

def print_stats(health, score):
    """Print the player's current stats."""
    print(f"Health: {health}, Score: {score}")

def print_position(position):
    """Print the player's current position."""
    print(f"You are at position {position}.")

def play_game():
    """Main game function."""
    size = 5  # Maze size (5x5)
    maze = generate_maze(size)
    position = (0, 0)  # Player starts at (0, 0)
    health = 100
    score = 0

    print("Welcome to Escape the Maze!")
    print("Navigate through the maze to reach the exit at position (4, 4).")
    print("Collect treasures to earn points and avoid traps to stay alive!")
    print("Commands: 'N' = North, 'S' = South, 'E' = East, 'W' = West.\n")

    while True:
        print_position(position)
        print_stats(health, score)

        # Check if player reached the exit
        if position == (size - 1, size - 1):
            print("Congratulations! You reached the exit!")
            print(f"Your final score is {score}.")
            break

        # Get player input
        move = input("Choose your action (N/S/E/W): ").strip().upper()

        # Calculate new position based on input
        new_position = position
        if move == "N":
            new_position = (position[0] - 1, position[1])
        elif move == "S":
            new_position = (position[0] + 1, position[1])
        elif move == "E":
            new_position = (position[0], position[1] + 1)
        elif move == "W":
            new_position = (position[0], position[1] - 1)
        else:
            print("Invalid move. Use 'N', 'S', 'E', or 'W'.")
            continue

        # Check if the move is valid (within maze boundaries)
        if new_position[0] < 0 or new_position[0] >= size or new_position[1] < 0 or new_position[1] >= size:
            print("You can't move outside the maze!")
            continue

        # Update the player's position
        position = new_position
        event = maze[position]

        # Handle events
        if event == "Treasure":
            print("You found a Treasure! +50 points.")
            score += 50
            maze[position] = "Empty"  # Clear the treasure
        elif event == "Trap":
            print("Oh no, it's a Trap! -20 health.")
            health -= 20
            maze[position] = "Empty"  # Clear the trap
        elif event == "Exit":
            print("You see the Exit! Keep going!")
        else:
            print("Nothing here. Keep moving!")

        # Check if the player is out of health
        if health <= 0:
            print("You ran out of health! Game over!")
            break

# Start the game
if __name__ == "__main__":
    play_game()
