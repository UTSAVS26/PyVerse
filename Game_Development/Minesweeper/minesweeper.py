import tkinter as tk
import random

class Minesweeper:
    def __init__(self, master, width=10, height=10, mines=10):
        self.master = master
        self.width = width
        self.height = height
        self.mines = mines
        self.buttons = []
        self.mines_positions = []
        self.game_over = False

        self.create_board()
        self.place_mines()
        self.calculate_numbers()

    def create_board(self):
        for row in range(self.height):
            button_row = []
            for col in range(self.width):
                button = tk.Button(self.master, text='', width=3, command=lambda r=row, c=col: self.reveal(r, c))
                button.grid(row=row, column=col)
                button_row.append(button)
            self.buttons.append(button_row)

    def place_mines(self):
        while len(self.mines_positions) < self.mines:
            pos = (random.randint(0, self.height - 1), random.randint(0, self.width - 1))
            if pos not in self.mines_positions:
                self.mines_positions.append(pos)

    def calculate_numbers(self):
        self.numbers = [[0 for _ in range(self.width)] for _ in range(self.height)]
        for (mine_row, mine_col) in self.mines_positions:
            for r in range(max(0, mine_row - 1), min(self.height, mine_row + 2)):
                for c in range(max(0, mine_col - 1), min(self.width, mine_col + 2)):
                    if (r, c) != (mine_row, mine_col):
                        self.numbers[r][c] += 1

    def reveal(self, row, col):
        if self.game_over:
            return

        if (row, col) in self.mines_positions:
            self.buttons[row][col].config(text='*', bg='red')
            self.game_over = True
            self.show_mines()
            return
        
        self.show_number(row, col)

    def show_number(self, row, col):
        if self.buttons[row][col]['text'] != '':
            return
        
        number = self.numbers[row][col]
        self.buttons[row][col].config(text=str(number), relief=tk.SUNKEN)

        if number == 0:
            for r in range(max(0, row - 1), min(self.height, row + 2)):
                for c in range(max(0, col - 1), min(self.width, col + 2)):
                    if (r, c) != (row, col):
                        self.reveal(r, c)

    def show_mines(self):
        for (mine_row, mine_col) in self.mines_positions:
            if self.buttons[mine_row][mine_col]['text'] == '':
                self.buttons[mine_row][mine_col].config(text='*', bg='yellow')

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Minesweeper")
    game = Minesweeper(root, width=10, height=10, mines=10)
    root.mainloop()

