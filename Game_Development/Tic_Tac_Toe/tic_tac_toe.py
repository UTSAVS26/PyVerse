import tkinter as tk
from tkinter import messagebox, simpledialog
import time
import threading
import winsound  # For Windows sound effects

class TicTacToe:
    def __init__(self, root):
        self.root = root
        self.root.title("Tic-Tac-Toe")
        self.board = [''] * 9
        self.current_player = 'X'
        self.buttons = []
        self.scores = {'X': 0, 'O': 0}
        self.player_names = {'X': "Player 1", 'O': "Player 2"}
        self.game_start_time = time.time()
        self.timer_label = tk.Label(self.root, text="Time: 0s", font='Arial 12')
        self.timer_label.grid(row=6, column=0, columnspan=3)
        self.running = True
        self.create_menu_bar()
        self.create_buttons()
        self.update_turn_label()
        self.update_score_label()
        self.start_timer()

    def create_menu_bar(self):
        menu_bar = tk.Menu(self.root)
        self.root.config(menu=menu_bar)
        game_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="Settings", menu=game_menu)
        game_menu.add_command(label="New Game", command=self.reset_board)
        game_menu.add_command(label="Change Players", command=self.change_players)
        game_menu.add_separator()
        game_menu.add_command(label="Exit", command=self.quit_game)

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

    def update_turn_label(self):
        self.turn_label = tk.Label(self.root, text=f"{self.player_names[self.current_player]}'s Turn ({self.current_player})", font='Arial 15')
        self.turn_label.grid(row=3, column=0, columnspan=3)

    def update_score_label(self):
        self.score_label = tk.Label(self.root, text=f"Score: {self.player_names['X']} (X) = {self.scores['X']} | {self.player_names['O']} (O) = {self.scores['O']}", font='Arial 12')
        self.score_label.grid(row=4, column=0, columnspan=3)

    def on_enter(self, event):
        button = event.widget
        if button['text'] == '':
            button['bg'] = '#D3D3D3'  # Light grey on hover

    def on_leave(self, event):
        button = event.widget
        if button['text'] == '':
            button['bg'] = 'lightgray'

    def make_move(self, idx):
        if self.board[idx] == '':
            self.play_sound("move")
            self.board[idx] = self.current_player
            button = self.buttons[idx // 3][idx % 3]
            button.config(text=self.current_player, state='disabled', disabledforeground='red' if self.current_player == 'X' else 'blue')

            winner_combination = self.check_winner()
            if winner_combination:
                self.highlight_winner(winner_combination)
                messagebox.showinfo("Game Over", f"{self.player_names[self.current_player]} ({self.current_player}) wins!")
                self.update_score(self.current_player)
                self.reset_board(keep_scores=True)
            elif '' not in self.board:
                messagebox.showinfo("Game Over", "It's a tie!")
                self.reset_board(keep_scores=True)
            else:
                self.switch_player()

    def switch_player(self):
        self.current_player = 'O' if self.current_player == 'X' else 'X'
        self.update_turn_label()

    def check_winner(self):
        winning_combinations = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # columns
            [0, 4, 8], [2, 4, 6]              # diagonals
        ]
        for combo in winning_combinations:
            if self.board[combo[0]] == self.board[combo[1]] == self.board[combo[2]] != '':
                return combo
        return None

    def highlight_winner(self, winner_combination):
        for idx in winner_combination:
            self.buttons[idx // 3][idx % 3].config(bg='green')

    def update_score(self, winner):
        self.scores[winner] += 1
        self.update_score_label()

    def reset_board(self, keep_scores=False):
        self.board = [''] * 9
        self.current_player = 'X'
        for row in self.buttons:
            for btn in row:
                btn.config(text='', state='normal', bg='lightgray')
        self.update_turn_label()
        if not keep_scores:
            self.scores = {'X': 0, 'O': 0}
            self.update_score_label()
        self.game_start_time = time.time()

    def change_players(self):
        x_name = simpledialog.askstring("Player X", "Enter Player X's name:", initialvalue=self.player_names['X'])
        o_name = simpledialog.askstring("Player O", "Enter Player O's name:", initialvalue=self.player_names['O'])
        if x_name:
            self.player_names['X'] = x_name
        if o_name:
            self.player_names['O'] = o_name
        self.update_turn_label()
        self.update_score_label()

    def start_timer(self):
        def update_time():
            while self.running:
                elapsed_time = int(time.time() - self.game_start_time)
                self.timer_label.config(text=f"Time: {elapsed_time}s")
                time.sleep(1)

        timer_thread = threading.Thread(target=update_time)
        timer_thread.daemon = True
        timer_thread.start()

    def quit_game(self):
        self.running = False
        self.root.quit()

    def play_sound(self, sound_type):
        if sound_type == "move":
            winsound.Beep(500, 100)

if __name__ == "__main__":
    root = tk.Tk()
    game = TicTacToe(root)
    root.mainloop()
