import tkinter as tk
from tkinter import messagebox
import random

class CrosswordPuzzle:
    def __init__(self, root):
        self.root = root
        self.root.title("Crossword Puzzle")
        
        # Directly embedding the words and hints
        self.words = [
            {"word": "REPRESENT", "hint": "To act or speak on behalf of someone or something."},
            {"word": "SOVEREIGN", "hint": "Possessing supreme or ultimate power."},
            {"word": "PARLIAMENT", "hint": "The supreme legislative body in a country."},
            {"word": "DEMOCRACY", "hint": "A system of government by the whole population."},
            {"word": "NECESSARY", "hint": "Required to be done, achieved, or present; essential."},
            {"word": "CHALLENGE", "hint": "A call to take part in a contest or competition."},
            {"word": "APPOINTED", "hint": "Assigned a position or role."},
            {"word": "DEVELOPED", "hint": "Having been made or grown over time."},
            {"word": "ELECTORAL", "hint": "Related to elections or voting."},
            {"word": "COMPETENT", "hint": "Having the necessary ability or skill."},
            {"word": "SIGNIFICANT", "hint": "Having meaning; important."},
            {"word": "PROFICIENT", "hint": "Skilled in a particular area."},
            {"word": "DEBILITATE", "hint": "To make someone weak or infirm."},
            {"word": "CONDITIONS", "hint": "The circumstances affecting a situation."},
            {"word": "INFORMANT", "hint": "A person who provides information."}
        ]
        
        self.current_level = 0
        self.shuffled_letters = []
        self.formed_word = ''
        self.selected_indexes = []
        
        self.create_widgets()
        self.update_buttons()

    def create_widgets(self):
        self.hint_label = tk.Label(self.root, text='', font=('Arial', 14))
        self.hint_label.pack(pady=10)

        self.grid_frame = tk.Frame(self.root)
        self.grid_frame.pack()

        self.letter_buttons = []
        for i in range(3):
            row_buttons = []
            for j in range(3):
                btn = tk.Button(self.grid_frame, text='', font=('Arial', 18), width=4, height=2,
                                command=lambda idx=i * 3 + j: self.select_letter(idx))
                btn.grid(row=i, column=j, padx=5, pady=5)
                row_buttons.append(btn)
            self.letter_buttons.append(row_buttons)

        self.formed_word_label = tk.Label(self.root, text='Formed Word: ', font=('Arial', 14))
        self.formed_word_label.pack(pady=10)

        self.reset_button = tk.Button(self.root, text='Reset', command=self.reset_game)
        self.reset_button.pack(side=tk.LEFT, padx=20)

        self.submit_button = tk.Button(self.root, text='Submit', command=self.check_word)
        self.submit_button.pack(side=tk.LEFT)

        self.show_hint()

    def show_hint(self):
        self.hint_label.config(text=f"Hint: {self.words[self.current_level]['hint']}")
        self.update_buttons()

    def update_buttons(self):
        # Shuffle the letters of the current word and update button texts
        self.shuffled_letters = random.sample(self.words[self.current_level]['word'], len(self.words[self.current_level]['word']))
        for i in range(3):
            for j in range(3):
                if i * 3 + j < len(self.shuffled_letters):
                    self.letter_buttons[i][j].config(text=self.shuffled_letters[i * 3 + j], state=tk.NORMAL)
                else:
                    self.letter_buttons[i][j].config(text='', state=tk.DISABLED)

    def select_letter(self, index):
        if index not in self.selected_indexes and index < len(self.shuffled_letters):
            self.selected_indexes.append(index)
            self.formed_word += self.shuffled_letters[index]
            self.formed_word_label.config(text=f'Formed Word: {self.formed_word}')

            # Disable the button to prevent re-selection
            btn_row = index // 3
            btn_col = index % 3
            self.letter_buttons[btn_row][btn_col].config(state=tk.DISABLED)

    def reset_game(self):
        self.formed_word = ''
        self.selected_indexes = []
        self.formed_word_label.config(text='Formed Word: ')
        self.update_buttons()

    def check_word(self):
        if self.formed_word == self.words[self.current_level]['word']:
            messagebox.showinfo("Correct!", f"Correct! {self.words[self.current_level]['word']}")
            self.move_to_next_level()
        else:
            messagebox.showerror("Oops!", "Try again.")
            self.reset_game()

    def move_to_next_level(self):
        if self.current_level < len(self.words) - 1:
            self.current_level += 1
        else:
            self.current_level = 0  # Restart from the first level if all are completed
        self.reset_game()
        self.show_hint()

if __name__ == "__main__":
    root = tk.Tk()
    app = CrosswordPuzzle(root)
    root.mainloop()
