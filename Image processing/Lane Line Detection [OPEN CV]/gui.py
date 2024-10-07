import tkinter as tk
from tkinter import *        # Importing all components from tkinter
import cv2                   # OpenCV for video processing
from PIL import Image, ImageTk  # PIL for converting OpenCV frames to images for Tkinter
import os
import numpy as np           # Numpy for working with arrays

# Global variables to store the last frames for both video captures
global last_frame1
last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)  # Placeholder for the first camera frame

global last_frame2
last_frame2 = np.zeros((480, 640, 3), dtype=np.uint8)  # Placeholder for the second camera frame

# Global video capture objects for two videos or cameras
global cap1
global cap2
cap1 = cv2.VideoCapture("./test2.mp4")  # Capturing from the first video file or camera
cap2 = cv2.VideoCapture("./test2.mp4")  # Capturing from the second video file or camera

# Function to display video from the first camera/video
def show_vid():                                       
    if not cap1.isOpened():                           # Check if the first camera is opened
        print("Can't open the camera1")               # Error handling if the camera can't be opened
    flag1, frame1 = cap1.read()                       # Read a frame from the first camera
    frame1 = cv2.resize(frame1, (600, 500))           # Resize the frame to 600x500
    if flag1 is None:                                 # Error handling if no frame is received
        print("Major error!")
    elif flag1:                                       # If a frame is received successfully
        global last_frame1
        last_frame1 = frame1.copy()                   # Store the frame in the global variable
        pic = cv2.cvtColor(last_frame1, cv2.COLOR_BGR2RGB)  # Convert BGR (OpenCV format) to RGB
        img = Image.fromarray(pic)                    # Convert the frame to a PIL image
        imgtk = ImageTk.PhotoImage(image=img)         # Convert the PIL image to Tkinter-compatible format
        lmain.imgtk = imgtk                           # Update the image on the label widget
        lmain.configure(image=imgtk)
        lmain.after(10, show_vid)                     # Call the function again after 10 ms to create a loop

# Function to display video from the second camera/video
def show_vid2():
    if not cap2.isOpened():                           # Check if the second camera is opened
        print("Can't open the camera2")               # Error handling if the camera can't be opened
    flag2, frame2 = cap2.read()                       # Read a frame from the second camera
    frame2 = cv2.resize(frame2, (600, 500))           # Resize the frame to 600x500
    if flag2 is None:                                 # Error handling if no frame is received
        print("Major error2!")
    elif flag2:                                       # If a frame is received successfully
        global last_frame2
        last_frame2 = frame2.copy()                   # Store the frame in the global variable
        pic2 = cv2.cvtColor(last_frame2, cv2.COLOR_BGR2RGB)  # Convert BGR (OpenCV format) to RGB
        img2 = Image.fromarray(pic2)                  # Convert the frame to a PIL image
        img2tk = ImageTk.PhotoImage(image=img2)       # Convert the PIL image to Tkinter-compatible format
        lmain2.img2tk = img2tk                        # Update the image on the label widget
        lmain2.configure(image=img2tk)
        lmain2.after(10, show_vid2)                   # Call the function again after 10 ms to create a loop

if __name__ == '__main__':
    root = tk.Tk()                                    # Create the root window for Tkinter
    heading = Label(root, image=img, text="Lane-Line Detection")  # Create a label with the title
    heading.pack()                                    # Add the label to the window
    heading2 = Label(root, text="Lane-Line Detection", pady=20, font=('arial', 45, 'bold'))  # Heading label
    heading2.configure(foreground='#364156')          # Set the text color
    heading2.pack()                                   # Add the heading label to the window

    # Create labels to display the video streams
    lmain = tk.Label(master=root)                     # Label for the first video
    lmain2 = tk.Label(master=root)                    # Label for the second video
    lmain.pack(side=LEFT)                             # Place the first video label on the left side
    lmain2.pack(side=RIGHT)                           # Place the second video label on the right side

    root.title("Lane-line Detection")                 # Set the window title
    root.geometry("1250x900+100+10")                  # Set the window size and position
    
    # Add a quit button to exit the application
    exitbutton = Button(root, text='Quit', fg="red", command=root.destroy).pack(side=BOTTOM)
    
    # Start showing the video streams
    show_vid()                                        # Start showing the first video
    show_vid2()                                       # Start showing the second video
    root.mainloop()                                   # Run the Tkinter main loop to keep the window open

    cap.release()                                     # Release the video capture when done
