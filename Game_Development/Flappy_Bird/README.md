# Flappy Bird Game

## 🎯 Goal
To create a fun and challenging Flappy Bird game clone with a graphical user interface using Python and Pygame.

## 🧾 Description
This project implements a Flappy Bird game, a side-scrolling game where the player controls a bird, attempting to fly between columns of pipes without hitting them. The game is built using Python's Pygame library, offering smooth animations and responsive controls.

## 🎮 Features
- Smooth bird animation and rotation
- Randomly generated pipes for endless gameplay
- Score tracking and high score saving
- Sound effects for enhanced gaming experience
- Clean and organized code structure for easy maintenance

## 🚀 Implementation
The game is implemented using:
- Pygame for game graphics and sound
- Python's built-in `random` module for generating pipe positions
- File I/O for saving and loading high scores

## 📚 Libraries Used
- pygame
- os
- random
- sys

## 📊 Usage
Ensure you have Python and Pygame installed. Then, run the script using Python:

```
python main.py
```

### How to Play:
- Press SPACE to make the bird flap and avoid pipes
- Try to pass through as many pipes as possible to increase your score
- If you hit a pipe or the ground, the game ends
- Press ENTER to restart the game after a game over

## 🛠️ Setup
1. Clone this repository or download the source code.
2. Ensure you have Python installed on your system.
3. Install Pygame by running: `pip install pygame`
4. Place all image and sound assets in the appropriate directories:
   - Images should be in a `sprites` folder
   - Sounds should be in an `audio` folder
5. Run the game using the command mentioned in the Usage section.

## 📁 File Structure
```
flappy_bird/
│
├── main.py
├── README.md
├── score.txt
├── sprites/
│   ├── background-day.png
│   ├── base.png
│   ├── bluebird-downflap.png
│   ├── bluebird-midflap.png
│   ├── bluebird-upflap.png
│   └── pipe-green.png
├── audio/
│   ├── die.wav
│   ├── hit.wav
│   ├── point.wav
│   └── wing.wav
└── favicon.ico
```

## 📢 Conclusion
This Flappy Bird clone provides an entertaining and challenging gaming experience. It demonstrates the use of Pygame for game development and showcases good practices in structuring game logic. The project is suitable for those learning game development with Python or for anyone looking to enjoy a classic arcade-style game.

## ✒️ Signature

**Aritro Saha**  
[GitHub](https://github.com/halcyon-past) | [LinkedIn](https://www.linkedin.com/in/aritro-saha/) | [Portfolio](https://aritro.tech/)