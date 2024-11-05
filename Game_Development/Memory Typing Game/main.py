import os
import time
import random

def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def calculate_accuracy(original, typed):
    """Calculate typing accuracy as a percentage"""
    if len(original) == 0:
        return 0
    
    correct = sum(1 for a, b in zip(original, typed) if a == b)
    return round((correct / len(original)) * 100, 2)

def get_random_text():
    """Return a random text for typing practice"""
    texts = [
        "The quick brown fox jumps over the lazy dog",
        "Programming is fun and rewarding",
        "Practice makes perfect",
        "Python is a versatile programming language",
        "Keep calm and keep coding",
        "Learning to type faster takes practice"
    ]
    return random.choice(texts)

def play_game():
    """Main game function"""
    score = 0
    rounds_played = 0
    
    while True:
        clear_screen()
        print("\n=== Memory Typing Game ===")
        print("\nRemember the text and type it exactly!")
        print("You'll have 3 seconds to memorize.")
        input("\nPress Enter when ready...")
        
        # Get random text and display it
        text_to_type = get_random_text()
        clear_screen()
        print("\nMemorize this text:")
        print(f"\n{text_to_type}")
        
        # Wait 3 seconds
        time.sleep(3)
        
        # Clear screen and get user input
        clear_screen()
        print("\nNow type the text:")
        user_input = input("\n> ")
        
        # Calculate accuracy
        accuracy = calculate_accuracy(text_to_type, user_input)
        
        # Update score
        rounds_played += 1
        if accuracy == 100:
            score += 1
        
        # Show results
        print(f"\nOriginal text: {text_to_type}")
        print(f"Your typing:   {user_input}")
        print(f"\nAccuracy: {accuracy}%")
        print(f"Perfect rounds: {score}/{rounds_played}")
        
        # Ask to play again
        play_again = input("\nPlay again? (y/n): ").lower()
        if play_again != 'y':
            break
    
    # Show final score
    print(f"\nFinal Score: {score}/{rounds_played} perfect rounds")
    print("Thanks for playing!")

if __name__ == "__main__":
    play_game()