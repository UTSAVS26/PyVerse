import turtle
import random

# Setup the screen
screen = turtle.Screen()
screen.title("Turtle Collection Game")
screen.setup(width=800, height=600)
screen.bgcolor("lightblue")

# Create player turtle
player = turtle.Turtle()
player.shape("turtle")
player.color("green")
player.penup()
player.speed(10)  # Slow animation speed to see the movement

# Create point turtle
point = turtle.Turtle()
point.shape("square")
point.color("gold")
point.penup()
point.speed(0)
point.hideturtle()  # Hide the point turtle initially

# Create obstacle turtles
obstacles = []
num_obstacles = 10
for _ in range(num_obstacles):
    obstacle = turtle.Turtle()
    obstacle.shape("square")
    obstacle.color("red")
    obstacle.penup()
    obstacle.speed(0)
    obstacle.goto(random.randint(-350, 350), random.randint(-250, 250))
    obstacles.append(obstacle)

# Score
score = 0

# Functions
def move_up():
    player.setheading(90)  # Move up

def move_down():
    player.setheading(270)  # Move down

def move_left():
    player.setheading(180)  # Move left

def move_right():
    player.setheading(0)  # Move right

def check_collision(t1, t2):
    return t1.distance(t2) < 20

def update_score():
    global score
    score += 10
    score_display.clear()
    score_display.write(f"Score: {score}", align="center", font=("Arial", 24, "normal"))

# Keyboard bindings
screen.listen()
screen.onkey(move_up, "Up")
screen.onkey(move_down, "Down")
screen.onkey(move_left, "Left")
screen.onkey(move_right, "Right")

# Display score
score_display = turtle.Turtle()
score_display.hideturtle()
score_display.penup()
score_display.goto(0, 260)
score_display.write(f"Score: {score}", align="center", font=("Arial", 24, "normal"))

# Place the point randomly
def place_point():
    x = random.randint(-350, 350)
    y = random.randint(-250, 250)
    point.goto(x, y)
    point.showturtle()

place_point()

# Game loop
while True:
    player.forward(1)  # Move the player turtle forward
    
    # Check collision with point
    if check_collision(player, point):
        update_score()
        place_point()
    
    # Check collision with obstacles
    for obstacle in obstacles:
        if check_collision(player, obstacle):
            score_display.clear()
            score_display.goto(0, 0)
            score_display.write("Game Over!", align="center", font=("Arial", 36, "bold"))
            turtle.done()
            break

    screen.update()  # Update the screen

screen.mainloop()
