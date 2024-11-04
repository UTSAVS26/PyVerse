import random

# Function to convert hex color to RGB values
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

# Function to play the Color Match Challenge
def play_color_match():
    # Randomly generate a hex color code
    hex_color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
    rgb_color = hex_to_rgb(hex_color)
    
    print("Welcome to the Color Match Challenge!")
    print(f"Your target color is: {hex_color}")

    attempts = 10
    while attempts > 0:
        print(f"\nAttempts remaining: {attempts}")
        
        # Get player guesses for Red, Green, and Blue
        try:
            guess_r = int(input("Enter your guess for Red (0-255): "))
            guess_g = int(input("Enter your guess for Green (0-255): "))
            guess_b = int(input("Enter your guess for Blue (0-255): "))
        except ValueError:
            print("Please enter valid numbers between 0 and 255.")
            continue
        
        # Provide feedback on the guesses
        if (guess_r, guess_g, guess_b) == rgb_color:
            print(f"Congratulations! You've matched the color {hex_color} correctly.")
            break
        else:
            if guess_r != rgb_color[0]:
                print(f"Red: {'Too high' if guess_r > rgb_color[0] else 'Too low'}")
            else:
                print("Red: Correct")
            
            if guess_g != rgb_color[1]:
                print(f"Green: {'Too high' if guess_g > rgb_color[1] else 'Too low'}")
            else:
                print("Green: Correct")
            
            if guess_b != rgb_color[2]:
                print(f"Blue: {'Too high' if guess_b > rgb_color[2] else 'Too low'}")
            else:
                print("Blue: Correct")
        
        attempts -= 1

    if attempts == 0:
        print(f"Game over! The correct RGB values for {hex_color} were {rgb_color}.")

# Main game loop
if __name__ == "__main__":
    while True:
        play_color_match()
        play_again = input("\nDo you want to play again? (yes/no): ").lower()
        if play_again != 'yes':
            print("Thank you for playing!")
            break