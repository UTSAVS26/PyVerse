from tkinter import *
from tkinter import messagebox
import random

class Medium:
    def user(self, color):  # Takes user choice
        self.color = color

    def __init__(self):
        self.colors_list = ['#270101', '#F08B33', '#776B04', '#F1B848', '#8F715B', '#0486DB', '#C1403D', '#F3D4A0']
        self.generated_colors = random.sample(self.colors_list, 4)

    def compare(self, user_choice):
        hints = []  # Generate hints based on user guess
        for i in range(4):
            if user_choice[i] == self.generated_colors[i]:
                hints.append('red')  # Correct color and position
            elif user_choice[i] in self.generated_colors:
                hints.append('gray')  # Correct color, wrong position
        return hints

class MasterMind:
    def __init__(self, root):
        self.root = root
        self.obj = Medium()
        self.generated_colors = self.obj.generated_colors
        self.colors_list = self.obj.colors_list
        root.geometry('400x600')
        root.title("MasterMind Game")
        
        # Configuring grid layout
        for y in range(20):
            Grid.rowconfigure(root, y, weight=1)
        for x in range(8):
            Grid.columnconfigure(root, x, weight=1)
        
        self.create_palette()
        self.user_guesses = []
        self.guess_code = []
        self.hint_labels = []

        global current_row, current_column
        current_column = 2
        current_row = 19

        # Remaining guesses
        self.remaining_guesses = 10

    def create_palette(self):
        # Creating the color palette for user to choose from
        self.palette_buttons = []
        for index, color in enumerate(self.colors_list):
            btn = Button(self.root, bg=color, height=2, width=5, relief=RAISED, command=lambda col=color: self.make_guess(col))
            btn.grid(row=20, column=index)
            self.palette_buttons.append(btn)

    def make_guess(self, chosen_color):
        global current_row, current_column
        if len(self.user_guesses) < 4:
            btn = Button(self.root, bg=chosen_color, height=2, width=5, relief=RAISED)
            btn.grid(row=current_row, column=current_column)
            self.guess_code.append(chosen_color)
            self.user_guesses.append(btn)
            current_column += 1

        if len(self.user_guesses) == 4:
            self.check_guess()
    
    def check_guess(self):
        global current_row, current_column
        # Compare guess with generated color code
        hints = self.obj.compare(self.guess_code)
        self.display_hints(hints)
        
        if hints == ['red', 'red', 'red', 'red']:
            self.display_message("Congratulations! You've cracked the code!")
            self.disable_palette()
        else:
            current_row -= 1
            current_column = 2
            self.guess_code = []
            self.user_guesses = []
            self.remaining_guesses -= 1
            
            if self.remaining_guesses == 0:
                self.display_message("Game Over! The correct code was: " + str(self.generated_colors))
                self.disable_palette()

    def display_hints(self, hints):
        global current_row
        hint_column = 6
        for hint in hints:
            hint_label = Label(self.root, bg=hint, width=2, height=1)
            hint_label.grid(row=current_row, column=hint_column)
            hint_column += 1

    def disable_palette(self):
        for btn in self.palette_buttons:
            btn.config(state=DISABLED)

    def display_message(self, message):
        messagebox.showinfo("Game Result", message)

# Running the MasterMind Game
root = Tk()
mastermind_game = MasterMind(root)
root.mainloop()
