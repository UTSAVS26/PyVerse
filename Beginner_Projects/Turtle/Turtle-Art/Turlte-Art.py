import turtle
import random
from turtle import Screen, Turtle

# Setup Turtle and Screen
screen = Screen()
screen.bgcolor("black")  # Background color set to black for better contrast

artist = Turtle()
artist.speed(0)  # Maximum speed for faster drawing
artist.hideturtle()  # Hide turtle icon
turtle.colormode(1.0)  # Use RGB colors

# Function to generate random colors
def random_color():
    return (random.random(), random.random(), random.random())

# Function to draw a square
def draw_square(size):
    for _ in range(4):
        artist.forward(size)
        artist.right(90)

# Function to draw a circle
def draw_circle(radius):
    artist.circle(radius)

# Function to draw a triangle
def draw_triangle(size):
    for _ in range(3):
        artist.forward(size)
        artist.left(120)

# Function to draw a star
def draw_star(size):
    for _ in range(5):
        artist.forward(size)
        artist.right(144)

# Function to draw random patterns with shapes
def draw_random_patterns(num_shapes):
    shapes = [draw_square, draw_circle, draw_triangle, draw_star]
    for _ in range(num_shapes):
        artist.penup()
        artist.goto(random.randint(-200, 200), random.randint(-200, 200))
        artist.pendown()
        artist.color(random_color())
        shape = random.choice(shapes)
        size_or_radius = random.randint(20, 100)
        shape(size_or_radius)

# Function to allow user customization
def customization_options():
    bg_color = input("Enter background color (e.g., black, white): ")
    screen.bgcolor(bg_color)

    while True:
        try:
            turtle_speed = int(input("Enter turtle speed (1-10): "))
            if 1 <= turtle_speed <= 10:
                artist.speed(turtle_speed)
                break
            else:
                print("Please enter a number between 1 and 10.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")

    while True:
        try:
            num_shapes = int(input("Enter the number of shapes to draw: "))
            if num_shapes > 0:
                draw_random_patterns(num_shapes)
                break
            else:
                print("Please enter a positive number.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")

# Execute the customization options
customization_options()

# Close the turtle graphics window when clicked
screen.exitonclick()