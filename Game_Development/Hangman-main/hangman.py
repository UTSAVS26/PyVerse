from tkinter import *
from tkinter import messagebox
from string import ascii_uppercase
import random
import time
import json  # For leaderboard persistence
from pygame import mixer  # For background music and sound effects

# Initialize the main window
window = Tk()
window.title('Enhanced Hangman - GUESS CITIES NAME')
window.geometry("900x600")
window.config(bg="lightblue")

# Word categories
categories = {
    "Indian Cities": [
        'MUMBAI', 'DELHI', 'BANGALORE', 'HYDERABAD', 'AHMEDABAD', 'CHENNAI', 'KOLKATA',
        'SURAT', 'PUNE', 'JAIPUR', 'AMRITSAR', 'ALLAHABAD', 'RANCHI', 'LUCKNOW', 'KANPUR',
        'NAGPUR', 'INDORE', 'THANE', 'BHOPAL', 'PATNA', 'GHAZIABAD', 'AGRA', 'FARIDABAD',
        'MEERUT', 'RAJKOT', 'VARANASI', 'SRINAGAR', 'RAIPUR', 'KOTA', 'JHANSI'
    ],
    "Fruits": ['APPLE', 'BANANA', 'MANGO', 'ORANGE', 'PEAR', 'PEACH', 'GRAPES', 'PAPAYA'],
    "Animals": ['TIGER', 'ELEPHANT', 'LION', 'MONKEY', 'ZEBRA', 'PANDA', 'KANGAROO'],
}

# Difficulty settings
DIFFICULTY_SETTINGS = {
    "Easy": 12,
    "Medium": 9,
    "Hard": 6
}

# Load images for hangman stages
photos = [PhotoImage(file=f"images/hang{i}.png") for i in range(12)]

# Initialize pygame mixer for music and sounds
mixer.init()
correct_sound = mixer.Sound("Correct.wav")
incorrect_sound = mixer.Sound("wrong.wav")

# Global variables
the_word_withSpaces = ""
numberOfGuesses = 0
score = 0
high_scores = {}
selected_category = "Indian Cities"
difficulty = "Easy"
timer = None
remaining_time = 0

# Load leaderboard data
try:
    with open("leaderboard.json", "r") as file:
        high_scores = json.load(file)
except FileNotFoundError:
    high_scores = {}

# Function to start a new game
def newGame():
    global the_word_withSpaces, numberOfGuesses, score, remaining_time, timer
    numberOfGuesses = 0
    score = 0
    update_score()

    the_word = random.choice(categories[selected_category])
    the_word_withSpaces = " ".join(the_word)
    lblWord.set(' '.join("_" * len(the_word)))
    imgLabel.config(image=photos[0])

    # Reset timer
    if timer:
        window.after_cancel(timer)
    remaining_time = DIFFICULTY_SETTINGS[difficulty] * 5
    update_timer()

# Timer functionality
def update_timer():
    global remaining_time, timer
    if remaining_time > 0:
        remaining_time -= 1
        lblTimer.set(f"Time Left: {remaining_time} seconds")
        timer = window.after(1000, update_timer)
    else:
        messagebox.showwarning("Time's Up!", "You ran out of time!")
        newGame()

# Function to guess a letter
def guess(letter):
    global numberOfGuesses, score
    if numberOfGuesses < DIFFICULTY_SETTINGS[difficulty]:
        txt = list(the_word_withSpaces)
        guessed = list(lblWord.get())
        if the_word_withSpaces.count(letter) > 0:
            correct_sound.play()
            for c in range(len(txt)):
                if txt[c] == letter:
                    guessed[c] = letter
            lblWord.set("".join(guessed))
            score += 10  # Increase score for correct guess
            update_score()

            if lblWord.get() == the_word_withSpaces:
                messagebox.showinfo("Hangman", "Congratulations! You guessed it!")
                save_score()
                newGame()
        else:
            incorrect_sound.play()
            numberOfGuesses += 1
            imgLabel.config(image=photos[numberOfGuesses])
            if numberOfGuesses == DIFFICULTY_SETTINGS[difficulty]:
                messagebox.showwarning("Hangman", "Game Over")
                save_score()
                newGame()

# Function to display the score
def update_score():
    lblScore.set(f"Score: {score}")

# Function to save high score
def save_score():
    global high_scores
    name = "Player"
    if name not in high_scores or score > high_scores[name]:
        high_scores[name] = score
        with open("leaderboard.json", "w") as file:
            json.dump(high_scores, file)

# Function to display leaderboard
def show_leaderboard():
    leaderboard_window = Toplevel(window)
    leaderboard_window.title("Leaderboard")
    leaderboard_window.geometry("400x300")
    sorted_scores = sorted(high_scores.items(), key=lambda x: x[1], reverse=True)

    Label(leaderboard_window, text="Leaderboard", font=("Helvetica", 18, "bold"), pady=10).pack()
    for idx, (player, score) in enumerate(sorted_scores[:10]):
        Label(leaderboard_window, text=f"{idx + 1}. {player}: {score}", font=("Helvetica", 14)).pack()

# Hint functionality
def use_hint():
    global numberOfGuesses, score
    if numberOfGuesses < DIFFICULTY_SETTINGS[difficulty]:
        hidden_letters = [i for i, ltr in enumerate(lblWord.get()) if ltr == "_"]
        if hidden_letters:
            random_index = random.choice(hidden_letters)
            guessed = list(lblWord.get())
            guessed[random_index] = the_word_withSpaces[random_index]
            lblWord.set("".join(guessed))
            numberOfGuesses += 1
            imgLabel.config(image=photos[numberOfGuesses])
            score -= 5  # Deduct score for using a hint
            update_score()
        else:
            messagebox.showinfo("Hint", "No more hints available!")

# UI Components
imgLabel = Label(window)
imgLabel.grid(row=0, column=0, columnspan=3, padx=10, pady=40)
lblWord = StringVar()
Label(window, textvariable=lblWord, font=('consolas', 24, 'bold'), bg="lightblue").grid(row=0, column=3, columnspan=6, padx=10)
lblScore = StringVar()
lblScore.set("Score: 0")
Label(window, textvariable=lblScore, font=('Helvetica', 14), bg="lightblue").grid(row=0, column=9, columnspan=2)
lblTimer = StringVar()
lblTimer.set("Time Left: 0 seconds")
Label(window, textvariable=lblTimer, font=("Helvetica", 14), bg="lightblue").grid(row=0, column=12, columnspan=2)

# Alphabet buttons
n = 0
for c in ascii_uppercase:
    Button(window, text=c, command=lambda c=c: guess(c), font=('Helvetica', 18), width=4).grid(row=1 + n // 9, column=n % 9)
    n += 1

# New Game, Hint, and Leaderboard buttons
Button(window, text="New Game", command=newGame, font=("Helvetica 10 bold")).grid(row=3, column=8)
Button(window, text="Hint", command=use_hint, font=("Helvetica 10 bold")).grid(row=4, column=8)
Button(window, text="Leaderboard", command=show_leaderboard, font=("Helvetica 10 bold")).grid(row=5, column=8)

# Category and Difficulty selection
category_var = StringVar(value="Indian Cities")
difficulty_var = StringVar(value="Easy")
OptionMenu(window, category_var, *categories.keys(), command=lambda x: set_category()).grid(row=6, column=8)
OptionMenu(window, difficulty_var, *DIFFICULTY_SETTINGS.keys(), command=lambda x: set_difficulty()).grid(row=7, column=8)

def set_category():
    global selected_category
    selected_category = category_var.get()
    newGame()

def set_difficulty():
    global difficulty
    difficulty = difficulty_var.get()
    newGame()

# Start the game
newGame()
window.mainloop()
