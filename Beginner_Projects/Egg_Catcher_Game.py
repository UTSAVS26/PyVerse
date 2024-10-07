from itertools import cycle
from random import randrange
from tkinter import Canvas, Tk, messagebox, font

class EggCatcherGame:
    def __init__(self):
        # Initialize game settings
        self.canvas_width = 800
        self.canvas_height = 400
        self.egg_width = 45
        self.egg_height = 55
        self.egg_score = 10
        self.egg_speed = 500
        self.egg_interval = 4000
        self.difficulty = 0.95
        self.catcher_width = 100
        self.catcher_height = 100
        self.score = 0
        self.lives_remaining = 3
        self.eggs = []

        # Initialize Tkinter root and canvas
        self.root = Tk()
        self.c = Canvas(self.root, width=self.canvas_width, height=self.canvas_height, background="deep sky blue")
        self.c.pack()

        # Draw the ground and the sun
        self.c.create_rectangle(-5, self.canvas_height-100, self.canvas_width+5, self.canvas_height+5, fill="sea green", width=0)
        self.c.create_oval(-80, -80, 120, 120, fill='orange', width=0)

        # Set up the color cycle for the eggs
        self.color_cycle = cycle(["light blue", "light green", "light pink", "light yellow", "light cyan"])

        # Create the catcher
        self.create_catcher()

        # Set up game font
        self.game_font = font.nametofont("TkFixedFont")
        self.game_font.config(size=18)

        # Create score and lives text
        self.score_text = self.c.create_text(10, 10, anchor="nw", font=self.game_font, fill="darkblue", text="Score: "+ str(self.score))
        self.lives_text = self.c.create_text(self.canvas_width-10, 10, anchor="ne", font=self.game_font, fill="darkblue", text="Lives: "+ str(self.lives_remaining))

        # Bind keys to catcher movement
        self.c.bind("<Left>", self.move_left)
        self.c.bind("<Right>", self.move_right)
        self.c.focus_set()

        # Start the game
        self.start_game()

    def create_catcher(self):
        # Create the catcher (arc shape)
        catcher_startx = self.canvas_width / 2 - self.catcher_width / 2
        catcher_starty = self.canvas_height - self.catcher_height - 20
        catcher_startx2 = catcher_startx + self.catcher_width
        catcher_starty2 = catcher_starty + self.catcher_height
        self.catcher = self.c.create_arc(catcher_startx, catcher_starty, catcher_startx2, catcher_starty2, start=200, extent=140, style="arc", outline="blue", width=3)

    def create_egg(self):
        # Create a new egg at a random position
        x = randrange(10, 740)
        y = 40
        new_egg = self.c.create_oval(x, y, x+self.egg_width, y+self.egg_height, fill=next(self.color_cycle), width=0)
        self.eggs.append(new_egg)
        self.root.after(self.egg_interval, self.create_egg)

    def move_eggs(self):
        # Move eggs down the screen
        for egg in self.eggs:
            (eggx, eggy, eggx2, eggy2) = self.c.coords(egg)
            self.c.move(egg, 0, 10)
            if eggy2 > self.canvas_height:
                self.egg_dropped(egg)
        self.root.after(self.egg_speed, self.move_eggs)

    def egg_dropped(self, egg):
        # Handle when an egg is dropped (missed)
        self.eggs.remove(egg)
        self.c.delete(egg)
        self.lose_a_life()
        if self.lives_remaining == 0:
            messagebox.showinfo("Game Over!", "Final Score: "+ str(self.score))
            self.root.destroy()

    def lose_a_life(self):
        # Decrease a life
        self.lives_remaining -= 1
        self.c.itemconfigure(self.lives_text, text="Lives: "+ str(self.lives_remaining))

    def check_catch(self):
        # Check if an egg is caught by the catcher
        (catcherx, catchery, catcherx2, catchery2) = self.c.coords(self.catcher)
        for egg in self.eggs:
            (eggx, eggy, eggx2, eggy2) = self.c.coords(egg)
            if catcherx < eggx and eggx2 < catcherx2 and catchery2 - eggy2 < 40:
                self.eggs.remove(egg)
                self.c.delete(egg)
                self.increase_score(self.egg_score)
        self.root.after(100, self.check_catch)

    def increase_score(self, points):
        # Increase the score
        self.score += points
        self.egg_speed = int(self.egg_speed * self.difficulty)
        self.egg_interval = int(self.egg_interval * self.difficulty)
        self.c.itemconfigure(self.score_text, text="Score: "+ str(self.score))

    def move_left(self, event):
        # Move the catcher to the left
        (x1, y1, x2, y2) = self.c.coords(self.catcher)
        if x1 > 0:
            self.c.move(self.catcher, -20, 0)

    def move_right(self, event):
        # Move the catcher to the right
        (x1, y1, x2, y2) = self.c.coords(self.catcher)
        if x2 < self.canvas_width:
            self.c.move(self.catcher, 20, 0)

    def start_game(self):
        # Start the game by creating the first egg, moving eggs, and checking for catches
        self.root.after(1000, self.create_egg)
        self.root.after(1000, self.move_eggs)
        self.root.after(1000, self.check_catch)
        self.root.mainloop()

# Create and start the game
if __name__ == "__main__":
    EggCatcherGame()
