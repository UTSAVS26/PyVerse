from itertools import cycle  # Used for cycling through colors for the eggs
from random import randrange  # Used to randomly place the eggs on the screen
from tkinter import Canvas, Tk, messagebox, font  # Tkinter components for the game GUI

# Set the canvas size
canvas_width = 800
canvas_height = 400

# Create the main window and canvas
root = Tk()
c = Canvas(root, width=canvas_width, height=canvas_height, background="deep sky blue")

# Draw the ground and the sun
c.create_rectangle(-5, canvas_height-100, canvas_width+5, canvas_height+5, fill="sea green", width=0)  # Green ground
c.create_oval(-80, -80, 120, 120, fill='orange', width=0)  # Orange sun
c.pack()

# Set up the color cycle for the eggs
color_cycle = cycle(["light blue", "light green", "light pink", "light yellow", "light cyan"])

# Egg properties
egg_width = 45
egg_height = 55
egg_score = 10  # Points per egg
egg_speed = 500  # Speed of egg movement (lower means faster)
egg_interval = 4000  # Interval for creating new eggs
difficulty = 0.95  # Difficulty increase factor (makes eggs fall faster over time)

# Catcher properties
catcher_color = "blue"
catcher_width = 100
catcher_height = 100

# Starting position of the catcher
catcher_startx = canvas_width / 2 - catcher_width / 2
catcher_starty = canvas_height - catcher_height - 20
catcher_startx2 = catcher_startx + catcher_width
catcher_starty2 = catcher_starty + catcher_height

# Create the catcher (arc shape)
catcher = c.create_arc(catcher_startx, catcher_starty, catcher_startx2, catcher_starty2, start=200, extent=140, style="arc", outline=catcher_color, width=3)

# Game font configuration
game_font = font.nametofont("TkFixedFont")
game_font.config(size=18)

# Score and lives setup
score = 0
score_text = c.create_text(10, 10, anchor="nw", font=game_font, fill="darkblue", text="Score: "+ str(score))

lives_remaining = 3
lives_text = c.create_text(canvas_width-10, 10, anchor="ne", font=game_font, fill="darkblue", text="Lives: "+ str(lives_remaining))

# List to hold active eggs
eggs = []

# Function to create a new egg at a random position
def create_egg():
    x = randrange(10, 740)  # Random x-coordinate for egg
    y = 40  # Egg starts at y=40
    new_egg = c.create_oval(x, y, x+egg_width, y+egg_height, fill=next(color_cycle), width=0)  # Create egg
    eggs.append(new_egg)  # Add egg to the list of eggs
    root.after(egg_interval, create_egg)  # Schedule the next egg creation

# Function to move eggs down the screen
def move_eggs():
    for egg in eggs:
        (eggx, eggy, eggx2, eggy2) = c.coords(egg)  # Get current egg coordinates
        c.move(egg, 0, 10)  # Move egg down by 10 pixels
        if eggy2 > canvas_height:  # If the egg goes off the screen
            egg_dropped(egg)  # Handle egg drop
    root.after(egg_speed, move_eggs)  # Schedule the next movement

# Function to handle when an egg is dropped (missed)
def egg_dropped(egg):
    eggs.remove(egg)  # Remove egg from the list
    c.delete(egg)  # Remove the egg from the canvas
    lose_a_life()  # Lose a life
    if lives_remaining == 0:  # If no lives left, game over
        messagebox.showinfo("Game Over!", "Final Score: "+ str(score))  # Show game over message
        root.destroy()  # Close the game window

# Function to decrease a life
def lose_a_life():
    global lives_remaining
    lives_remaining -= 1  # Decrease the life count
    c.itemconfigure(lives_text, text="Lives: "+ str(lives_remaining))  # Update the lives display

# Function to check if an egg is caught by the catcher
def check_catch():
    (catcherx, catchery, catcherx2, catchery2) = c.coords(catcher)  # Get catcher coordinates
    for egg in eggs:
        (eggx, eggy, eggx2, eggy2) = c.coords(egg)  # Get egg coordinates
        if catcherx < eggx and eggx2 < catcherx2 and catchery2 - eggy2 < 40:  # If the egg is caught
            eggs.remove(egg)  # Remove the egg
            c.delete(egg)  # Delete the egg from the canvas
            increase_score(egg_score)  # Increase the score
    root.after(100, check_catch)  # Schedule the next catch check

# Function to increase the score
def increase_score(points):
    global score, egg_speed, egg_interval
    score += points  # Increase the score
    egg_speed = int(egg_speed * difficulty)  # Increase the speed of the eggs
    egg_interval = int(egg_interval * difficulty)  # Decrease the interval between eggs
    c.itemconfigure(score_text, text="Score: "+ str(score))  # Update the score display

# Function to move the catcher to the left
def move_left(event):
    (x1, y1, x2, y2) = c.coords(catcher)  # Get catcher coordinates
    if x1 > 0:  # Check if catcher is not at the left edge
        c.move(catcher, -20, 0)  # Move catcher left by 20 pixels

# Function to move the catcher to the right
def move_right(event):
    (x1, y1, x2, y2) = c.coords(catcher)  # Get catcher coordinates
    if x2 < canvas_width:  # Check if catcher is not at the right edge
        c.move(catcher, 20, 0)  # Move catcher right by 20 pixels

# Bind the left and right arrow keys to move the catcher
c.bind("<Left>", move_left)
c.bind("<Right>", move_right)

# Set focus on the canvas to capture key events
c.focus_set()

# Start the game by creating the first egg, moving eggs, and checking for catches
root.after(1000, create_egg)
root.after(1000, move_eggs)
root.after(1000, check_catch)

# Run the Tkinter main loop (start the game window)
root.mainloop()
