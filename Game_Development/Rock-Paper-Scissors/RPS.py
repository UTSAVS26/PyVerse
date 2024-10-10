import tkinter as tk
from tkinter import PhotoImage
import random
from PIL import Image, ImageTk  # Pillow library for image handling

class RockPaperScissorsGame:
    def __init__(self, master):
        self.master = master
        self.master.title("Rock Paper Scissors")
        self.master.geometry("500x600")

        # Load background image
        self.background_image = Image.open("C:\\Users\\ASUS\\all python files\\rpsGame\\background.png")  # Use your image file path
        self.background_image = self.background_image.resize((500, 600), Image.ANTIALIAS)
        self.bg_image = ImageTk.PhotoImage(self.background_image)

        self.bg_label = tk.Label(master, image=self.bg_image)
        self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)

        # Load icons
        self.rock_icon = ImageTk.PhotoImage(Image.open("C:\\Users\\ASUS\\all python files\\rpsGame\\rock.png").resize((100, 100)))  # Replace with your icons
        self.paper_icon = ImageTk.PhotoImage(Image.open("C:\\Users\\ASUS\\all python files\\rpsGame\\paper.png").resize((100, 100)))
        self.scissors_icon = ImageTk.PhotoImage(Image.open("C:\\Users\\ASUS\\all python files\\rpsGame\\scissors.png").resize((100, 100)))

        # Scores
        self.user_score = 0
        self.computer_score = 0

        # Score Label
        self.score_label = tk.Label(master, text="Score: You - 0, Computer - 0", font=("Arial", 14), bg="#ffffff")
        self.score_label.pack(pady=20)

        # Result Label
        self.result_label = tk.Label(master, text="", font=("Arial", 16), bg="#ffffff")
        self.result_label.pack(pady=10)

        # Buttons for Rock, Paper, Scissors
        self.buttons_frame = tk.Frame(master, bg="#ffffff")
        self.buttons_frame.pack(pady=30)

        self.rock_button = tk.Button(self.buttons_frame, image=self.rock_icon, command=lambda: self.play("rock"),
                                     bd=0, bg="#ffffff", activebackground="#ffffff")
        self.rock_button.grid(row=0, column=0, padx=20, pady=10)
        self.paper_button = tk.Button(self.buttons_frame, image=self.paper_icon, command=lambda: self.play("paper"),
                                      bd=0, bg="#ffffff", activebackground="#ffffff")
        self.paper_button.grid(row=0, column=1, padx=20, pady=10)
        self.scissors_button = tk.Button(self.buttons_frame, image=self.scissors_icon, command=lambda: self.play("scissors"),
                                         bd=0, bg="#ffffff", activebackground="#ffffff")
        self.scissors_button.grid(row=0, column=2, padx=20, pady=10)

        # Restart Button
        self.restart_button = tk.Button(master, text="Restart", command=self.restart_game, font=("Arial", 12), bg="#FFDDC1", activebackground="#FFC857")
        self.restart_button.pack(pady=10)

        # Hover effect
        self.rock_button.bind("<Enter>", lambda event, b=self.rock_button: self.on_enter(b))
        self.rock_button.bind("<Leave>", lambda event, b=self.rock_button: self.on_leave(b))
        self.paper_button.bind("<Enter>", lambda event, b=self.paper_button: self.on_enter(b))
        self.paper_button.bind("<Leave>", lambda event, b=self.paper_button: self.on_leave(b))
        self.scissors_button.bind("<Enter>", lambda event, b=self.scissors_button: self.on_enter(b))
        self.scissors_button.bind("<Leave>", lambda event, b=self.scissors_button: self.on_leave(b))

    def play(self, user_choice):
        computer_choice = random.choice(["rock", "paper", "scissors"])
        result = self.determine_winner(user_choice, computer_choice)
        self.update_scores(result)
        self.display_result(user_choice, computer_choice, result)

    def determine_winner(self, user_choice, computer_choice):
        if user_choice == computer_choice:
            return "It's a tie!"
        elif (user_choice == "rock" and computer_choice == "scissors") or \
             (user_choice == "scissors" and computer_choice == "paper") or \
             (user_choice == "paper" and computer_choice == "rock"):
            return "You win!"
        else:
            return "You lose!"

    def update_scores(self, result):
        if result == "You win!":
            self.user_score += 1
        elif result == "You lose!":
            self.computer_score += 1

    def display_result(self, user_choice, computer_choice, result):
        self.score_label.config(text=f"Score: You - {self.user_score}, Computer - {self.computer_score}")
        self.result_label.config(text=f"You chose: {user_choice}, Computer chose: {computer_choice}\n{result}")

    def restart_game(self):
        self.user_score = 0
        self.computer_score = 0
        self.score_label.config(text="Score: You - 0, Computer - 0")
        self.result_label.config(text="")

    # Hover effects for buttons
    def on_enter(self, button):
        button.config(bg="#FFC857")  # Highlight color

    def on_leave(self, button):
        button.config(bg="#ffffff")  # Return to default

if __name__ == "__main__":
    root = tk.Tk()
    game = RockPaperScissorsGame(root)
    root.mainloop()
