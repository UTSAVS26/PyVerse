# Memory Typing Game ⌨️

This is an engaging **Python-based command-line** implementation of a **Memory Typing Game**. Test and improve your typing skills by memorizing and reproducing displayed text within a time limit. Challenge yourself with different difficulty levels and various text categories!

---

## 🎯 Objective

The goal is to accurately type the displayed text after it disappears from the screen. The game tests both your memory and typing speed while helping you improve your typing accuracy and recall abilities.

---

## 🚀 How to Play

1. **Start the Game**: Launch the Python script to begin.
2. **Choose Difficulty**: Select your preferred difficulty level:
   - Easy: Shorter texts, longer display time (10 seconds)
   - Medium: Moderate length texts, standard display time (7 seconds)
   - Hard: Longer texts, shorter display time (5 seconds)
3. **Memorize Text**: A text snippet will appear for a limited time.
4. **Type from Memory**: After the text disappears, type what you remember.
5. **View Results**: Receive feedback on:
   - Typing accuracy (%)
   - Words per minute (WPM)
   - Time taken
6. **Progress**: Track your improvement over multiple rounds.

---

## 🛠 System Requirements

- **Operating System**: Any system running Python 3.x
- **Python Version**: Python 3.x or higher

### Dependencies

```bash
pip install colorama   # For colored terminal output
pip install difflib    # For text comparison
```

---

## 🔧 How to Run

1. Clone the repository and navigate to the project folder:
   ```bash
   git clone <repository-url>
   cd memory-typing-game
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the game:
   ```bash
   python3 memory_typing_game.py
   ```

---

## 📚 Game Mechanics

- **Text Generation**: 
  - Random selection from various categories (quotes, facts, sentences)
  - Difficulty-appropriate length selection
  - No repetition within same session

- **Scoring System**:
  - Accuracy: Calculated using difflib sequence matcher
  - Speed: Words per minute calculation
  - Time Bonus: Extra points for quick completion
  - Perfect Match Bonus: Additional points for 100% accuracy

- **Performance Tracking**:
  - Session high scores
  - Personal best records
  - Progress statistics

---

## 💻 System Specifications

- Python Version: 3.x+
- Required Space: < 10MB
- Memory Usage: Minimal (~50MB)
- Terminal: Any terminal with Unicode support

---

## 📖 Features

### Core Features
- Multiple difficulty levels
- Various text categories
- Real-time typing feedback
- Performance statistics
- Progress tracking

### Game Modes
1. **Classic Mode**: Standard memory typing challenge
2. **Time Attack**: Reduced display time for each successful round
3. **Marathon**: Consecutive challenges with increasing difficulty
4. **Practice Mode**: No time limit, focus on accuracy

---

## 🤔 Tips for Success

1. **Memory Techniques**:
   - Break text into meaningful chunks
   - Create mental associations
   - Focus on key words first

2. **Typing Tips**:
   - Maintain proper hand positioning
   - Focus on accuracy over speed
   - Practice regular finger exercises

3. **Strategy**:
   - Start with easier levels
   - Gradually increase difficulty
   - Take short breaks between rounds
   - Review mistakes after each attempt

---

## 🔄 Future Updates

- Online leaderboard
- Custom text categories
- Multiplayer mode
- Advanced statistics
- Achievement system
- Custom difficulty settings

---

## 🐛 Troubleshooting

- **Display Issues**: Ensure terminal supports Unicode
- **Performance Lag**: Close resource-heavy applications
- **Text Not Showing**: Check terminal color support
- **Score Not Saving**: Verify write permissions

---

## 📝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

---

Enjoy improving your typing and memory skills with the Memory Typing Game! 🎯✨