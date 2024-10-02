**Turtle Graphics**

## Overview
The Turtle module in Python is a beginner-friendly graphics library that allows users to create complex shapes, patterns, and designs by giving simple commands to a "turtle" on the screen. It's a great way to learn the basics of programming, while making coding visually engaging and fun.

## Installation
The Turtle module comes pre-installed with Python, so there's no need for additional installations. However, you can ensure it’s available by running:

     pip install PythonTurtle


## Features:
1) Direction:
 Turtle can move a specified number of steps forward and backward.
 - Move forward by x steps: 
     ```sh
     forward(20)  
     ``` 
 - Move backward by x steps: 
     ```sh
     backward(20)  
     ```     
2) Turn: 
 Turtle can turn by given degree in either direction (right or left).
     ```sh
     right(45)  
     ```
3) Shape: 
 You can decide the shape of turtle using shape() function.
 Available shapes are: turtle, circle, square, triangle, arrow, classic.
     ```sh
     shape("arrow")
     ```
4) Color:
 You can control the color of turtle's drawing and appearance.
 - To set drawing color:
     ```sh 
     color('red')
     ```
 - To set pen color:
     ```sh
     pencolor('green')
     ```    
 - To set background color:
     ```sh
     bgcolor('black')
     ```    
5) Speed:
 You can control the turtle's speed with speed().
     ```sh 
     speed(10)
     ```
6) Turtle position:
 - Send turtle back to starting point.   
     ```sh
     home()
     ```  
    Mainly used when turtle go off-screen.
 - Get turtle's current position co-ordinates.
     ```sh
     pos()
     ```
 - Clear all previous drawing, i.e. to get a clear screen.
     ```sh
     clearscreen()
     ```
7) Pen Control: 
 You can lift the pen up and down to control when the turtle draws.
 - Lift the pen (Stop drawing):
     When user want to move turtle without drawing anything.
     ```sh
     up()
     ```
 - Put pen down (Start drawing):
     When user want to start drawing again.
     ```sh
     down()
     ```
8) Width:
 You can set the width of the turtle’s drawing line with width(x), where x is an integer value.
     ```sh 
     width(x)
     ```


## General Turtle Graphics Code Syntax

1) Import the Turtle Module.
2) Set up the screen.
3) Create a function to draw desired shape or pattern.
4) Take user input (if required) and pass it to the function.
5) Run the program.

Don’t forget to use **turtle.done()** to keep the window open after the drawing is complete.

## Example: Spiral Pattern

For refernece, you can check out this [Spiral Patten example](turtle_spiral.py) created using turtle module.

## Further Reading
For more detailed usage of the Turtle module, visit the official [Python documentation](https://docs.python.org/3/library/turtle.html).


