import pyautogui
import numpy as np
import cv2
import keyboard
import tkinter as tk


def handle_start(event):
     # create a video writer object
    resolution = (1920, 1080)

    # specify how to compress & encode the video
    code = cv2.VideoWriter_fourcc(*"XVID")

    # name the output 
    name = "ScreenRec.avi"

    # specify frame rate, will experiment with it to obtain the best result
    fps = 60

    obj = cv2.VideoWriter(name, code, fps, resolution)


    while True:
        # take a ss
        ss = pyautogui.screenshot()
        
        # convert the screenshot to a numpy array
        frame = np.array(ss)
        
         # default opencv capture is in bgr color scheme (inverted)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
        # write it to the output file
            
        obj.write(frame)
            
        if keyboard.is_pressed('q'):
            break 

    # discard video writer object and close windows   
    cv2.destroyAllWindows

    obj.release()

window = tk.Tk()
greeting = tk.Label(text="Welcome to ScreenRec. \n Once you start recording, press the close button to stop. Would you like to-", width=60, height=20, bg="magenta")
greeting.pack(fill=tk.X)
btn = tk.Button(text="Start", bg="green", fg="white")
btn.bind("<Button-1>", handle_start)
btn.pack(fill=tk.X)
window.mainloop()
print("done")

    
    
