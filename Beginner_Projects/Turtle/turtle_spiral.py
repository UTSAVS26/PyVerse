import turtle as t

# Setup the display window where turtle will draw
window = t.Screen()
# Configure Screen size
window.setup(800,800)
#set the speed of turtle
t.speed(50)
# Set background color
t.bgcolor('black')

# Define a function to draw spiral with  total steps and list of colors
def spiral(steps,color_list):
    for step in range(steps):
        for c in color_list:
            t.width(step/50 )       # Set pattern width 
            t.color(c)              # Set's turtle color
            t.forward(step)         # Move the turtle forward by "steps"
            t.left(30)              # Turn the turtle 30 degree to left
            
if __name__ == '__main__':
    print("Sprial printing!!")
    total_steps = int(input("enter no. of steps: "))
    # split(',') denotes split input of colors string into list
    color_list = input("enter the list of colors separated by commas: ").split(',')   
    spiral(total_steps,color_list) 
    t.done()

          
            