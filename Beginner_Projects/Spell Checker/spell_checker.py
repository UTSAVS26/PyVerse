import tkinter as tk
from tkinter import messagebox

# Load words from words.txt into a set for fast lookup
def load_words():
    with open("words.txt", "r") as file:
        words = set(word.strip().lower() for word in file.readlines())
    return words

# Check if the word is spelled correctly
def is_word_correct(word, word_list):
    return word.lower() in word_list

# Suggest corrections for the misspelled word
def suggest_corrections(word, word_list):
    suggestions = []
    for w in word_list:
        if abs(len(w) - len(word)) <= 1:  # Find words of similar lengths
            suggestions.append(w)
    return suggestions

# Check spelling when the button is pressed
def check_spelling():
    input_word = word_entry.get()
    if not input_word:
        messagebox.showwarning("Input Error", "Please enter a word.")
        return
    
    if is_word_correct(input_word, word_list):
        result_label.config(text=f"'{input_word}' is spelled correctly!", fg="green")
    else:
        result_label.config(text=f"'{input_word}' is not spelled correctly.", fg="red")
        suggestions = suggest_corrections(input_word, word_list)
        if suggestions:
            result_label.config(text=f"'{input_word}' is not spelled correctly.\nDid you mean: {', '.join(suggestions[:5])}?", fg="red")

# Load words list
word_list = load_words()

# Initialize the Tkinter GUI window
root = tk.Tk()
root.title("Spell Checker")

# Input Label
word_label = tk.Label(root, text="Enter a word:")
word_label.pack(pady=10)

# Entry widget for user to input a word
word_entry = tk.Entry(root, width=40)
word_entry.pack(pady=5)

# Button to check spelling
check_button = tk.Button(root, text="Check Spelling", command=check_spelling)
check_button.pack(pady=10)

# Label to display the result
result_label = tk.Label(root, text="", font=("Helvetica", 14))
result_label.pack(pady=20)

# Start the Tkinter event loop
root.mainloop()
