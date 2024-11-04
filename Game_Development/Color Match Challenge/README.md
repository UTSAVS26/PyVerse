# Color Match Challenge 🎨

This is a **Python-based command-line game** where the player must guess the correct RGB (Red, Green, Blue) values for a randomly generated hex color code. It's an engaging way to test your knowledge of color codes and learn how hex values correspond to RGB values.

## 🎯 Objective

The objective of the game is to match a randomly generated hex color code (e.g., `#FF5733`) with its correct RGB values. You have a limited number of attempts to guess the values for Red, Green, and Blue, each ranging from 0 to 255.

## 🚀 How to Play

1. **Start the Game**: Run the Python script to begin.
2. **Receive a Hex Color Code**: A random hex color code will be displayed (e.g., `#AABBCC`).
3. **Enter Your Guesses**: Input your guesses for the Red, Green, and Blue values.
4. **Get Feedback**: After each guess, the game will tell you if your guesses are:
   - **Too high** or **Too low** for each RGB value.
   - **Correct** when your guess matches the exact RGB value.
5. **Keep Guessing**: You have a limited number of attempts to guess the correct values.
6. **End the Game**: The game will either end with success when you match all the RGB values or failure if you run out of attempts.
7. **Play Again**: After the game ends, you will be asked if you want to play again.


## 🛠 System Requirements

- **Operating System**: Any system that can run Python 3.x
- **Python Version**: Python 3.x or higher

## 🔧 How to Run

1. Clone the repository and navigate to the project folder:

   ```bash
   git clone <repository-url>
   cd <project-folder>
   ```

2. Run the Python script:

   ```bash
   python3 main.py
   ```

3. Follow the instructions to play the game.

## 📚 Game Mechanics

- **Hex to RGB Conversion**: The game converts a randomly generated hex color code into its corresponding RGB values.
- **Player Input**: The player guesses the RGB values through a series of inputs.
- **Feedback Mechanism**: The game provides feedback on whether each guess is too high, too low, or correct.
- **Game Loop**: The player keeps guessing until they either match the correct RGB values or run out of attempts.

## 💡 Strategy Tips

1. Start your guesses around the middle value (128) and adjust based on feedback.
2. Use the "too high" and "too low" hints to narrow down your guesses for each RGB component.
3. Pay close attention to the feedback and use a process of elimination to match the color quickly.

## 🏆 Game Features

- **Random Color Generation**: Each game features a randomly generated hex color.
- **Limited Attempts**: You have a finite number of attempts to guess the correct values.
- **Replay Option**: After each game, you can choose to play again or exit.
- **Educational**: This game is a great way to practice and learn more about RGB color codes.

Enjoy playing the **Color Match Challenge** and sharpen your RGB guessing skills! 🎉