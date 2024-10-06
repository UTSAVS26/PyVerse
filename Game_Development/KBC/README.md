
# KBC Quiz Game 🎮

## 🎯 Objective

This Python code creates a quiz game modeled after Kaun Banega Crorepati (KBC), where players answer multiple-choice questions to win increasing amounts of money. Here's an overview of the key parts of the code:

---

**Modules Used**:

* `random`: For selecting random questions and answers.
* `time`: For adding slight delays between actions to make the game experience more dynamic.

**Game Introduction**:

* The game starts with a welcome message and prompts the player to enter their name.

**Questions and Answers**:

* A list of `questions` and corresponding `answers` is provided.
* Each question also has a list of wrong answers (`wronganswers`) for generating multiple-choice options.

**Question Loop**:

* The game continuously presents questions to the player until they either answer incorrectly or choose to quit.
* For each question, a random question is selected from the list, along with random wrong answers and one correct answer.
* The answer options are then shuffled before being displayed to the player.

**Winning Amount**:

* The player earns progressively higher amounts with each correct answer, starting from ₹1,000 and going up to ₹1,500,000.
* If the player answers all 10 questions correctly, they win the game and the grand prize.

**Game Termination:**

* The game can end in three ways:
* The player answers incorrectly.
* The player quits by pressing "Q".
* The player answers all 10 questions correctly and wins the maximum amount.

**Feedback:**

* The game provides immediate feedback on whether the player's answer was correct or wrong, and displays their current winnings after each round.

---

## 🚀 How to Play Kaun Banega Crorepati (KBC) Python Quiz Game

1. **Start the Game**: Run the program to begin the quiz. 
2. **Answering Questions**: 
* The game will present you with a series of multiple-choice questions (MCQs).
* Each question will have four options (A, B, C, D). Only one option is the correct answer.
* To answer, simply type the letter of the correct option (e.g., a, b, c, or d) and press Enter.

3. **Winning Amounts**: 
* For each correct answer, you will win a certain amount of money.
* The amount increases as you answer more questions correctly
4. **Quitting the Game**: 
* If you wish to quit at any point and take home your winnings, you can press the letter q when prompted for an answer.
* You will receive the total amount you’ve earned up until that point.
5. **Game Over - Wrong Answer**: 
* If you provide a wrong answer, the game will end.
* Your winnings will be displayed, and you will not proceed to the next question.
6. **Winning the Game**: 
* If you answer all 10 questions correctly, you will win the grand prize of ₹15,00,000!
* Congratulations will be displayed, and you’ll be crowned the winner of the game.

---

## 🔧 How to Run

1. Clone the repository and navigate to the project folder:
   ```bash
   git clone <repository-url>
   cd <project-folder>
   ```

2. Install dependencies (if any) and then run the program:
   ```bash
   python3 KBCquiz.py
   ```

3. Enjoy playing the KBC game!

---

