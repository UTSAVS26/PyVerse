import tkinter as tk
from tkinter import messagebox

root = tk.Tk()
root.title("Tic-Tac-Toe")

player = 'X'

# Create a 3x3 grid of buttons
buttons = [[None, None, None], [None, None, None], [None, None, None]]

# This list will track the state of the board
board_state = [['' for _ in range(3)] for _ in range(3)]

# Function to check for a winner
def check_winner():
    # Check rows, columns, and diagonals
    for i in range(3):
        if board_state[i][0] == board_state[i][1] == board_state[i][2] != '':
            return board_state[i][0]
        if board_state[0][i] == board_state[1][i] == board_state[2][i] != '':
            return board_state[0][i]
    
    if board_state[0][0] == board_state[1][1] == board_state[2][2] != '':
        return board_state[0][0]
    if board_state[0][2] == board_state[1][1] == board_state[2][0] != '':
        return board_state[0][2]
    
    return None

# Function to handle button clicks
def button_click(row, col):
    global player
    if board_state[row][col] == '' and buttons[row][col]["text"] == "":
        buttons[row][col]["text"] = player
        board_state[row][col] = player
        winner = check_winner()
        if winner:
            messagebox.showinfo("Tic-Tac-Toe", f"Player {winner} wins!")
            reset_game()
        elif all(board_state[r][c] != '' for r in range(3) for c in range(3)):
            messagebox.showinfo("Tic-Tac-Toe", "It's a tie!")
            reset_game()
        else:
            # Switch player
            player = 'O' if player == 'X' else 'X'

# Function to reset the game
def reset_game():
    global player
    player = 'X'
    for r in range(3):
        for c in range(3):
            buttons[r][c]["text"] = ""
            board_state[r][c] = ''

# Create and place the buttons in the grid
for row in range(3):
    for col in range(3):
        buttons[row][col] = tk.Button(root, text="", font=("Helvetica", 20), height=3, width=6,
                                      command=lambda row=row, col=col: button_click(row, col))
        buttons[row][col].grid(row=row, column=col)

reset_button = tk.Button(root, text="Reset", font=("Helvetica", 14), command=reset_game)
reset_button.grid(row=3, column=0, columnspan=3)

root.mainloop()
