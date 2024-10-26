from tkinter import *
from tkinter import messagebox
from string import ascii_uppercase
import random

# Initialize the main window
window = Tk()
window.title('Enhanced Hangman - GUESS CITIES NAME')
window.geometry("700x500")
window.config(bg="lightblue")

# Word list
word_list = [
    'MUMBAI', 'DELHI', 'BANGALORE', 'HYDERABAD', 'AHMEDABAD', 'CHENNAI', 'KOLKATA', 
    'SURAT', 'PUNE', 'JAIPUR', 'AMRITSAR', 'ALLAHABAD', 'RANCHI', 'LUCKNOW', 'KANPUR',
    'NAGPUR', 'INDORE', 'THANE', 'BHOPAL', 'PATNA', 'GHAZIABAD', 'AGRA', 'FARIDABAD',
    'MEERUT', 'RAJKOT', 'VARANASI', 'SRINAGAR', 'RAIPUR', 'KOTA', 'JHANSI'
]

# Load images for hangman stages
photos = [PhotoImage(file=f"images/hang{i}.png") for i in range(12)]

# Global variables
the_word_withSpaces = ""
numberOfGuesses = 0
score = 0

# Function to start a new game
def newGame():
    global the_word_withSpaces, numberOfGuesses, score
    numberOfGuesses = 0
    score = 0
    update_score()
    
    the_word = random.choice(word_list)
    the_word_withSpaces = " ".join(the_word)
    lblWord.set(' '.join("_" * len(the_word)))
    imgLabel.config(image=photos[0])

# Function to guess a letter
def guess(letter):
    global numberOfGuesses, score
    if numberOfGuesses < 11:
        txt = list(the_word_withSpaces)
        guessed = list(lblWord.get())
        if the_word_withSpaces.count(letter) > 0:
            for c in range(len(txt)):
                if txt[c] == letter:
                    guessed[c] = letter
            lblWord.set("".join(guessed))
            score += 10  # Increase score for correct guess
            update_score()
            
            if lblWord.get() == the_word_withSpaces:
                messagebox.showinfo("Hangman", "Congratulations! You guessed it!")
                newGame()
        else:
            numberOfGuesses += 1
            imgLabel.config(image=photos[numberOfGuesses])
            if numberOfGuesses == 11:
                messagebox.showwarning("Hangman", "Game Over")
                newGame()
                
# Function to display the score
def update_score():
    lblScore.set(f"Score: {score}")

# Hint functionality
def use_hint():
    global numberOfGuesses, score
    if numberOfGuesses < 11:
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

# Alphabet buttons
n = 0
for c in ascii_uppercase:
    Button(window, text=c, command=lambda c=c: guess(c), font=('Helvetica', 18), width=4).grid(row=1 + n // 9, column=n % 9)
    n += 1

# New Game and Hint buttons
Button(window, text="New Game", command=newGame, font=("Helvetica 10 bold")).grid(row=3, column=8)
Button(window, text="Hint", command=use_hint, font=("Helvetica 10 bold")).grid(row=4, column=8)

# Start the game
newGame()
window.mainloop()
