import tkinter as tk
from tkinter import messagebox

class TicTacToe:
    def __init__(self, root):
        self.root = root
        self.root.title("Tic-Tac-Toe")
        self.board = [''] * 9
        self.current_player = 'X'
        self.buttons = []
        self.scores = {'X': 0, 'O': 0}  # Scoreboard for both players

        # Display current player's turn
        self.turn_label = tk.Label(self.root, text=f"Player {self.current_player}'s Turn", font='Arial 15')
        self.turn_label.grid(row=3, column=0, columnspan=3)

        # Add Menu Bar
        self.create_menu_bar()

        # Display score
        self.score_label = tk.Label(self.root, text=f"Score: X = {self.scores['X']} | O = {self.scores['O']}", font='Arial 12')
        self.score_label.grid(row=4, column=0, columnspan=3)

        # Reset Game Button
        self.reset_button = tk.Button(self.root, text="Reset Game", font='Arial 12', command=self.reset_board)
        self.reset_button.grid(row=5, column=0, columnspan=3)

        self.create_buttons()

    def create_menu_bar(self):
        menu_bar = tk.Menu(self.root)
        self.root.config(menu=menu_bar)
        game_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="Settings", menu=game_menu)

        # Add New Game option to the menu
        game_menu.add_command(label="New Game", command=self.reset_board)

        # Add Exit option to the menu
        game_menu.add_separator()
        game_menu.add_command(label="Exit", command=self.root.quit)

    def create_buttons(self):
        for i in range(3):
            row = []
            for j in range(3):
                btn = tk.Button(self.root, text='', font='Arial 20', width=5, height=2,
                                command=lambda idx=i*3+j: self.make_move(idx))
                btn.grid(row=i, column=j)
                btn.bind("<Enter>", self.on_enter)
                btn.bind("<Leave>", self.on_leave)
                row.append(btn)
            self.buttons.append(row)

    def on_enter(self, event):
        button = event.widget
        if button['text'] == '':
            button['bg'] = '#D3D3D3'  # Light grey on hover when empty

    def on_leave(self, event):
        button = event.widget
        if button['text'] == '':
            button['bg'] = 'lightgray'  # Reset to normal color when leaving

    def make_move(self, idx):
        if self.board[idx] == '':
            self.board[idx] = self.current_player
            button = self.buttons[idx // 3][idx % 3]
            button.config(text=self.current_player, state='disabled', disabledforeground='red' if self.current_player == 'X' else 'blue')


            # Check for winner
            winner_combination = self.check_winner()
            if winner_combination:
                self.highlight_winner(winner_combination)  #Highlight the winning line
                messagebox.showinfo("Game Over", f"Player {self.current_player} wins!")
                self.update_score(self.current_player)
                self.reset_board()  #Reset board after highlighting the winner
            elif '' not in self.board:
                messagebox.showinfo("Game Over", "It's a tie!")
                self.reset_board()
            else:
                self.current_player = 'O' if self.current_player == 'X' else 'X'
                self.turn_label.config(text=f"Player {self.current_player}'s Turn")

    def check_winner(self):
        winning_combinations = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # columns
            [0, 4, 8], [2, 4, 6]              # diagonals
        ]
        for combo in winning_combinations:
            if self.board[combo[0]] == self.board[combo[1]] == self.board[combo[2]] != '':
                return combo  #Return the winning combination
        return None

    def highlight_winner(self, winner_combination):
        #Highlight the winning combination by changing the background color
        for idx in winner_combination:
            self.buttons[idx // 3][idx % 3].config(bg='green')

    def update_score(self, winner):
        self.scores[winner] += 1
        self.score_label.config(text=f"Score: X = {self.scores['X']} | O = {self.scores['O']}")

    def reset_board(self):
        self.board = [''] * 9
        self.current_player = 'X'
        for row in self.buttons:
            for btn in row:
                btn.config(text='', state='normal', bg='lightgray')
        self.turn_label.config(text=f"Player {self.current_player}'s Turn")

if __name__ == "__main__":
    root = tk.Tk()
    game = TicTacToe(root)
    root.mainloop()
