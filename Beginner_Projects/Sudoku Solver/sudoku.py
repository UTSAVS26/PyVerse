import tkinter as tk
from tkinter import messagebox

def is_valid(board, row, col, num):
    """Checks if placing num in board[row][col] is valid."""
    # Check the row
    for c in range(6):
        if board[row][c] == num:
            return False
    # Check the column
    for r in range(6):
        if board[r][col] == num:
            return False
    # Check the 2x3 subgrid
    start_row = (row // 2) * 2
    start_col = (col // 3) * 3
    for r in range(start_row, start_row + 2):
        for c in range(start_col, start_col + 3):
            if board[r][c] == num:
                return False
    return True

def solve_sudoku(board):
    """Solves the Sudoku puzzle using backtracking."""
    for row in range(6):
        for col in range(6):
            if board[row][col] == 0:
                for num in range(1, 7):  # Numbers 1-6 for 6x6 Sudoku
                    if is_valid(board, row, col, num):
                        board[row][col] = num
                        if solve_sudoku(board):
                            return True
                        board[row][col] = 0
                return False
    return True

def get_board():
    """Retrieve the current board from the input fields."""
    board = []
    for r in range(6):
        row = []
        for c in range(6):
            val = entries[r][c].get()
            row.append(int(val) if val.isdigit() and 1 <= int(val) <= 6 else 0)
        board.append(row)
    return board

def solve():
    """Solve the Sudoku puzzle and update the GUI."""
    board = get_board()
    if solve_sudoku(board):
        for r in range(6):
            for c in range(6):
                if board[r][c] != 0:
                    entries[r][c].delete(0, tk.END)
                    entries[r][c].insert(0, str(board[r][c]))
    else:
        messagebox.showinfo("No Solution", "No solution exists for the given Sudoku puzzle.")

# Create the main window
root = tk.Tk()
root.title("6x6 Sudoku Solver")

# Create a 6x6 grid of entry fields
entries = []
for r in range(6):
    row_entries = []
    for c in range(6):
        entry = tk.Entry(root, width=2, font=('Arial', 24), borderwidth=2, relief="solid")
        entry.grid(row=r, column=c, padx=5, pady=5)
        row_entries.append(entry)
    entries.append(row_entries)

# Create a solve button
solve_button = tk.Button(root, text="Solve", font=('Arial', 16), command=solve)
solve_button.grid(row=6, column=0, columnspan=6)

# Start the GUI event loop
root.mainloop()

