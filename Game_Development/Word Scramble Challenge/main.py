import random
import time

# List of words for the Word Scramble Challenge
words_list = ["python", "developer", "challenge", "scramble", "puzzle", "algorithm", "function", "variable"]

# Function to scramble a word
def scramble_word(word):
    scrambled = list(word)  # Convert the word into a list of characters
    random.shuffle(scrambled)  # Shuffle the characters in the list
    return ''.join(scrambled)  # Join the shuffled characters back into a string

# Function to play the Word Scramble Challenge
def play_game():
    score = 0  # Player's initial score
    hints_used = 0  # Track how many hints the player used
    time_limit = 60  # Time limit for the game (in seconds)

    start_time = time.time()  # Record the start time of the game

    # Play the game until the time limit is reached
    while time.time() - start_time < time_limit:
        # Select a random word from the list and scramble it
        word = random.choice(words_list)
        scrambled_word = scramble_word(word)
        
        print(f"\nScrambled Word: {scrambled_word}")
        
        # Get the player's input
        guess = input("Your guess: ").lower()

        # Check if the player wants a hint
        if guess == "hint":
            print(f"Hint: The first letter is '{word[0]}'")
            hints_used += 1
            continue

        # Check if the guess is correct
        if guess == word:
            print("Correct!")
            score += 10  # Increase the score for a correct answer
        else:
            print(f"Incorrect! The correct word was: {word}")
        
        # Check if time is up
        if time.time() - start_time >= time_limit:
            print("\nTime's up!")
            break

    # Display the player's final score
    print(f"\nGame Over! Your final score is: {score}")
    if hints_used > 0:
        print(f"Hints used: {hints_used} (Hint penalty applied)")
    
# Main program starts here
if __name__ == "__main__":
    print("Welcome to the Word Scramble Challenge!")
    print("Unscramble the word or type 'hint' for a clue (but it will reduce your score).")
    print("You have 60 seconds. Let's begin!")

    play_game()  # Start the game