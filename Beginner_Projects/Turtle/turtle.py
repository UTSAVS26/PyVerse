import turtle as t

# Setup the display window where turtle will draw
window = t.Screen()
window.setup(800, 800)  # Configure Screen size
t.speed(50)              # Set the speed of turtle
t.bgcolor('black')       # Set background color

# Define a function to draw a spiral with a specified number of steps, a list of colors, and a direction
def spiral(steps, color_list, angle, direction):
    for step in range(steps):
        for c in color_list:
            t.width(step / 50)        # Set pen width
            t.color(c)                # Set turtle color
            t.forward(step)           # Move the turtle forward by "steps"
            if direction == 'left':
                t.left(angle)         # Turn the turtle left
            else:
                t.right(angle)        # Turn the turtle right

def get_user_input():
    while True:
        try:
            total_steps = int(input("Enter the number of steps: "))
            if total_steps <= 0:
                raise ValueError("Number of steps must be positive.")
            break
        except ValueError as e:
            print(f"Invalid input: {e}. Please try again.")

    color_list = input("Enter the list of colors separated by commas: ").split(',')
    color_list = [color.strip() for color in color_list if color.strip()]  # Clean up whitespace

    if not color_list:
        print("You must provide at least one color.")
        return None, None, None, None

    while True:
        try:
            angle = int(input("Enter the angle for turning (suggestion: 30): "))
            break
        except ValueError:
            print("Invalid input. Please enter a valid integer for the angle.")

    while True:
        direction = input("Enter the direction to turn (left/right): ").strip().lower()
        if direction in ['left', 'right']:
            break
        else:
            print("Invalid input. Please enter 'left' or 'right'.")

    return total_steps, color_list, angle, direction

if __name__ == '__main__':
    print("Spiral printing!!")
    total_steps, color_list, angle, direction = get_user_input()

    if total_steps is not None and color_list is not None:
        spiral(total_steps, color_list, angle, direction)
        t.done()
