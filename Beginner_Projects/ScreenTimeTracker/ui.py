import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import analyzer
import tkinter as tk
from PIL import Image, ImageTk
import os

def show_heatmap(root):
    path = max([os.path.join("assets", f) for f in os.listdir("assets")], key=os.path.getctime)
    img = Image.open(path).resize((600, 300))
    img_tk = ImageTk.PhotoImage(img)

    label = tk.Label(root, image=img_tk)
    label.image = img_tk  # Keep a reference
    label.pack()

def launch_ui():
    root = tk.Tk()
    root.title("ScreenTimeTracker")
    root.geometry("600x450")

    style = ttk.Style()
    style.theme_use("clam")

    ttk.Label(root, text="Screen Time Tracker", font=("Helvetica", 20)).pack(pady=10)

    def show_chart():
        analyzer.generate_chart()
        img = Image.open("assets/pie_chart.png")
        img = ImageTk.PhotoImage(img)
        chart_panel.config(image=img)
        chart_panel.image = img

    def start_tracking():
        status_label.config(text="Tracking started...")

    def stop_tracking():
        status_label.config(text="Tracking stopped.")

    ttk.Button(root, text="Generate Pie Chart", command=show_chart).pack(pady=10)
    chart_panel = ttk.Label(root)
    chart_panel.pack(pady=10)
    tk.Button(root, text="Start", command=start_tracking, bg="#4CAF50", fg="white").pack(pady=10)
    tk.Button(root, text="Stop", command=stop_tracking, bg="#f44336", fg="white").pack(pady=10)

    status_label = tk.Label(root, text="", font=("Arial", 10))
    status_label.pack(pady=10)

    root.mainloop()