import tkinter as tk
from tkinter import messagebox

class TicTacToe:
    def __init__(self, root):
        self.root = root
        self.root.title("Tic-Tac-Toe")
        self.board = [''] * 9
        self.current_player = 'X'
        self.buttons = []

        self.create_buttons()
    
    def create_buttons(self):
        for i in range(3):
            row = []
            for j in range(3):
                btn = tk.Button(self.root, text='', font='Arial 20', width=5, height=2,
                                command=lambda idx=i*3+j: self.make_move(idx))
                btn.grid(row=i, column=j)
                row.append(btn)
            self.buttons.append(row)

    def make_move(self, idx):
        if self.board[idx] == '':
            self.board[idx] = self.current_player
            self.buttons[idx // 3][idx % 3].config(text=self.current_player)
            if self.check_winner():
                messagebox.showinfo("Game Over", f"Player {self.current_player} wins!")
                self.reset_game()
            elif '' not in self.board:
                messagebox.showinfo("Game Over", "It's a tie!")
                self.reset_game()
            else:
                self.current_player = 'O' if self.current_player == 'X' else 'X'

    def check_winner(self):
        winning_combinations = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # columns
            [0, 4, 8], [2, 4, 6]               # diagonals
        ]
        for combo in winning_combinations:
            if self.board[combo[0]] == self.board[combo[1]] == self.board[combo[2]] != '':
                return True
        return False

    def reset_game(self):
        self.board = [''] * 9
        self.current_player = 'X'
        for row in self.buttons:
            for btn in row:
                btn.config(text='')

if __name__ == "__main__":
    root = tk.Tk()
    game = TicTacToe(root)
    root.mainloop()

