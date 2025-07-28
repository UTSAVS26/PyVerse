import tkinter as tk
from tkinter import ttk
from datetime import datetime
import csv
import os
import matplotlib.pyplot as plt

# 🗂️ Ensure folders exist
os.makedirs("logs", exist_ok=True)
os.makedirs("assets", exist_ok=True)

log_file = "logs/log.csv"
if not os.path.exists(log_file):
    with open(log_file, "w", newline="") as f:
        csv.writer(f).writerow(["Date", "Start Time", "End Time", "Duration (seconds)"])

start_time = None

def start_tracking():
    global start_time
    start_time = datetime.now()
    status_label.config(text=f"🟢 Tracking started at {start_time.strftime('%H:%M:%S')}")

def stop_tracking():
    if start_time:
        end_time = datetime.now()
        duration = (end_time - start_time).seconds

        with open(log_file, "a", newline="") as f:
            csv.writer(f).writerow([
                start_time.strftime("%Y-%m-%d"),
                start_time.strftime("%H:%M:%S"),
                end_time.strftime("%H:%M:%S"),
                duration
            ])

        generate_pie_chart()

        summary = (
            f"✅ Session Recorded\n"
            f"📅 {start_time.strftime('%Y-%m-%d')}\n"
            f"⏰ {start_time.strftime('%H:%M:%S')} ➡ {end_time.strftime('%H:%M:%S')}\n"
            f"🕒 Duration: {duration} sec"
        )
        status_label.config(text=summary)
    else:
        status_label.config(text="⚠️ Start tracking first!")

def generate_pie_chart():
    usage_by_day = {}
    with open(log_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            date = row["Date"]
            duration = int(row["Duration (seconds)"])
            usage_by_day[date] = usage_by_day.get(date, 0) + duration

    labels = list(usage_by_day.keys())
    sizes = list(usage_by_day.values())

    plt.figure(figsize=(5, 5))
    plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140)
    plt.title("📊 Screen Time by Date")
    chart_path = f"assets/screentime_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(chart_path)
    plt.close()

# 🎨 UI Setup
root = tk.Tk()
root.title("📱 ScreenTime Tracker")
root.geometry("520x420")
root.configure(bg="#e3f2fd")

# Font Styling
font_title = ("Segoe UI", 20, "bold")
font_button = ("Segoe UI", 12)
font_status = ("Segoe UI", 12)

tk.Label(root, text="ScreenTime Tracker", font=font_title, bg="#e3f2fd", fg="#1e88e5").pack(pady=(20, 10))

ttk.Style().configure("TButton", font=font_button, padding=10)
ttk.Button(root, text="▶️ Start Tracking", command=start_tracking).pack(pady=10)
ttk.Button(root, text="⏹️ Stop Tracking", command=stop_tracking).pack(pady=10)

status_label = tk.Label(root, text="", font=font_status, bg="#e3f2fd", fg="#37474f", justify="center")
status_label.pack(pady=30)

tk.Label(root, text="📈 Pie chart saved to assets/", font=("Segoe UI", 10), bg="#e3f2fd", fg="#616161").pack(pady=5)

root.mainloop()