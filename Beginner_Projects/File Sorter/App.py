import os
import csv
import shutil 

import tkinter as tk

from tkinter import *
from tkinter import ttk
from tkinter.ttk import *

from datetime import datetime

import tkinter.font as tkFont

from tkinter import filedialog

files_organized = False

moved_files = []

def log_file_movement(log_file, time, source_file, target_folder):
  with open(log_file, mode='a', newline='') as file:
    log_writer = csv.writer(file)
    log_writer.writerow([time, source_file, target_folder])

def undo():
    for (source, target) in moved_files:
      shutil.move(target, source)
      print_text(f"Undo: Moved {target} back to {source}")
      print_text("Undo")
    global files_organized

    files_organized = False
    undo_button.config(state=tk.DISABLED)

def organize_files():
  
  directory = directory_entry.get()
  file_extensions = extensions_entry.get().split(',')
  file_extensions = [ext.strip() for ext in file_extensions if ext]

  log_file = log_entry.get()
  
  if not os.path.exists(directory):
    print_text("Directory does not exist.")
    return

  try:
    if log_file:
      log_file_exists = os.path.exists(log_file)
      with open(log_file, mode='a', newline='') as file:
        log_writer = csv.writer(file)
        if not log_file_exists:
          log_writer.writerow(["Time", "Source File", "Target Folder"])

    for root, dirs, files in os.walk(directory):
      for file in files:
        file_extension = os.path.splitext(file)[1]
        if not file_extensions or file_extension in file_extensions:
          target_subdirectory = os.path.join(directory, file_extension[1:].upper() + " Files")

          if not os.path.exists(target_subdirectory):
            os.mkdir(target_subdirectory)

          source_path = os.path.join(root, file)
          target_path = os.path.join(target_subdirectory, file)

          try:
            shutil.move(source_path, target_path)
            print_text(f"Moved {file} to {target_subdirectory}")

            if log_file:
              log_file_movement(log_file, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), file, target_subdirectory)

            moved_files.append((source_path, target_path))
          except Exception as e:
            print_text(f"Error moving {file}: {str(e)}")

    print_text("File organization completed")

    global files_organized
    files_organized = True

    if files_organized:
        undo_button.config(state=tk.NORMAL)

  except KeyboardInterrupt:
    print_text("\nFile organization interrupted. Starting the undo process...")
    undo()
    print_text("Undo completed. No files are moved.")
    return

def change_theme():
    
    global window

    if window.tk.call("ttk::style", "theme", "use") == "azure-dark":

        window.tk.call("set_theme", "light")

    else:

        window.tk.call("set_theme", "dark")

# Tkinter GUI Code

window = tk.Tk()
window.title("File Organizer")

height = 500
width = 600

window.geometry(f"{width}x{height}")

font = "Consolas"

window.resizable(width=True, height=True)

window.tk.call("source", r"Theme\azure.tcl")
window.tk.call("set_theme", "light")

text = tk.Text(window)

roots_frame = tk.LabelFrame(window, labelanchor="n", bg="white")
roots_frame.place(x= 25,y=300, width=550, height= 150)

def print_text(text):
      
      global roots_frame

      Label(roots_frame, text=text,font=('Consolas 10'), fg="black").pack()


app_label = ttk.Label(window, text = "File Sorter", font= "Consolas 30")
app_label.place(y=10, x=180)

directory_label = ttk.Label(window, text="Directory Path", font=font)
directory_label.place(y=85, x=10)

directory_entry = ttk.Entry(window)
directory_entry.place(y=80, x=200, width=200)

browse_button = ttk.Button(window, text="Browse", command=lambda: [directory_entry.insert(0, filedialog.askdirectory())])
browse_button.place(y=80, x=450)

extensions_label = ttk.Label(window, text="File Extensions", font = font)
extensions_label.place(y=130, x=10)

extensions_entry = ttk.Entry(window)
extensions_entry.place(y=125, x=200, width=200)

log_label = ttk.Label(window, text="Log File", font = font)
log_label.place(y=180, x=10)

log_entry = ttk.Entry(window) 
log_entry.place(y=175, x=200, width=200)

organize_button = ttk.Button(window, text="Organize Files", command=organize_files)
organize_button.place(y=220, x=200, width=110)

undo_button = ttk.Button(window, text="Undo", command=undo, state=tk.DISABLED)
undo_button.place(y=220, x=330, width=70)

button = ttk.Button(window, text="Theme", command=change_theme, compound="left" )
button.place(y = 220, x = 450)

window.mainloop()
