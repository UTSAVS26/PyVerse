import tkinter as tk
from tkinter import messagebox
from random import randint

#Configuration 
MAX_SCORE = 20  
BG_COLOR = "#1f1f2e"
FG_COLOR = "#f0f0f0"
ACCENT = "#6ee7b7"
FONT = ("Helvetica", 14, "bold")
SCORE_FONT = ("Helvetica", 12)
BUTTON_STYLE = {"font": FONT, "bg": "#333", "fg": FG_COLOR, "activebackground": "#444", "bd": 0, "width": 12}

#Varibales
player_scores = []
current_player = 0
num_players = 0
score_labels = []


root = tk.Tk()
root.title("🎲 Dice Game - Don't Roll a 1!")
root.configure(bg=BG_COLOR)

# Layout
frame = tk.Frame(root, bg=BG_COLOR)
frame.pack(padx=30, pady=30)

status_label = tk.Label(frame, text="Welcome to Dice Game!", font=("Helvetica", 18, "bold"), fg=ACCENT, bg=BG_COLOR)
status_label.pack(pady=10)
status_label = tk.Label(frame, text="First to roll a 20 wins!", font=("Helvetica", 18, "bold"), fg=ACCENT, bg=BG_COLOR)
status_label.pack(pady=10)


#Dice roll 
dice_label = tk.Label(frame, text="", font=("Helvetica", 30), fg=ACCENT, bg=BG_COLOR)
dice_label.pack(pady=10)

#Score
score_frame = tk.Frame(frame, bg=BG_COLOR)
score_frame.pack(pady=10)

#Logic
def roll_dice():
    global current_player

    dice = randint(1, 6)
    dice_label.config(text=f"🎲 {dice}")
    if dice == 1:
        player_scores[current_player] = 0
        update_scores()
        messagebox.showinfo("Oops!", f"Player {current_player + 1} rolled a 1! Score resets to 0.")
        next_turn()
    else:
        player_scores[current_player] += dice
        update_scores()
        if player_scores[current_player] >= MAX_SCORE:
            status_label.config(text=f"🏆 Player {current_player + 1} Wins!", fg="#facc15")
            dice_label.config(text="🎉")
            roll_button.pack_forget()
            pass_button.pack_forget()
            return

def pass_turn():
    next_turn()

def next_turn():
    global current_player
    current_player = (current_player + 1) % num_players
    update_turn_display()

def update_scores():
    for i in range(num_players):
        score_labels[i].config(
            text=f"Player {i + 1}: {player_scores[i]} pts",
            fg=ACCENT if i == current_player else FG_COLOR
        )

def update_turn_display():
    status_label.config(text=f"Player {current_player + 1}'s Turn", fg="#93c5fd")
    update_scores()
    dice_label.config(text="")

def start_game():
    global num_players, player_scores, score_labels, current_player

    try:
        num_players = int(player_entry.get())
        if not 2 <= num_players <= 4:
            raise ValueError
    except ValueError:
        messagebox.showerror("Invalid Input", "Enter a valid number of players (2-4).")
        return

    player_entry_frame.pack_forget()
    player_scores = [0] * num_players
    current_player = 0
    score_labels.clear()

    #Clear old labels if replayed
    for widget in score_frame.winfo_children():
        widget.destroy()

    for i in range(num_players):
        label = tk.Label(score_frame, text=f"Player {i + 1}: 0 pts", font=SCORE_FONT, bg=BG_COLOR)
        label.pack()
        score_labels.append(label)

    roll_button.pack(pady=5)
    pass_button.pack(pady=5)
    update_turn_display()

#Player Entry UI
player_entry_frame = tk.Frame(frame, bg=BG_COLOR)
player_entry_frame.pack(pady=20)

tk.Label(player_entry_frame, text="How many players? (2–4)", font=FONT, fg=FG_COLOR, bg=BG_COLOR).pack(side="left", padx=5)
player_entry = tk.Entry(player_entry_frame, width=5, font=FONT)
player_entry.pack(side="left", padx=5)
tk.Button(player_entry_frame, text="🎮 Start", command=start_game, **BUTTON_STYLE).pack(side="left", padx=5)

#Roll and pass buttons
roll_button = tk.Button(frame, text="🎲 Roll", command=roll_dice, **BUTTON_STYLE)
pass_button = tk.Button(frame, text="⏭️ Pass", command=pass_turn, **BUTTON_STYLE)

#Run App
root.mainloop()
