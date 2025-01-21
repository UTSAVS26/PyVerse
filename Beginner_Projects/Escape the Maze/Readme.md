# Escape the Maze

**Escape the Maze** is a beginner-friendly, text-based adventure game built in Python. The goal is to navigate through a randomly generated maze, collect treasures, avoid traps, and reach the exit before running out of health.

---

## Features

1. **Random Maze Generation**  
   - A unique 5x5 maze is generated for each game.

2. **Player Movement**  
   - Navigate using simple commands:  
     `N` (North), `S` (South), `E` (East), and `W` (West).

3. **Treasures and Traps**  
   - **Treasures**: Increase your score (+50 points).  
   - **Traps**: Decrease your health (-20 health).

4. **Win or Lose Conditions**  
   - **Win**: Reach the exit (4, 4) with health remaining.  
   - **Lose**: Run out of health before reaching the exit.

5. **Beginner-Friendly Code**  
   - Perfect for those learning Python. Covers basic concepts like loops, conditionals, dictionaries, and randomization.

---

## How to Play

1. Clone this repository to your local machine.  
2. Run the game using Python on your terminal.  
3. Use the following commands to navigate through the maze:
   - `N`: Move North (Up)
   - `S`: Move South (Down)
   - `E`: Move East (Right)
   - `W`: Move West  
4. Collect treasures, avoid traps, and reach the exit at position `(4, 4)` before running out of health.

---

## Example Gameplay

```plaintext
Welcome to Escape the Maze!
Navigate through the maze to reach the exit at position (4, 4).
Commands: 'N' = North, 'S' = South, 'E' = East, 'W' = West.

You are at position (0, 0).
Health: 100, Score: 0
Choose your action (N/S/E/W): E
You moved East to position (0, 1).
You found a Treasure! +50 points.

You are at position (0, 1).
Health: 100, Score: 50
Choose your action (N/S/E/W): S
You moved South to position (1, 1).
Oh no, it's a Trap! -20 health.

...

Congratulations! You reached the exit!
Your final score is 150.
```

---

## Contributions

Contributions are welcome! If you have ideas to enhance the game, feel free to raise an issue or submit a pull request.

Some ideas for enhancements:
- Increase maze size (e.g., 10x10 grid).  
- Add power-ups (e.g., health packs or extra treasures).  
- Implement a graphical version using `pygame`.

---

## License

This project is licensed under the [MIT License](LICENSE).
