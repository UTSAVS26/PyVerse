import tkinter as tk
from tkinter import messagebox
import pandas as pd
import time
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageTk

# Paths
LOG_PATH = "logs/log.csv"
ASSETS_PATH = "assets"
os.makedirs("logs", exist_ok=True)
os.makedirs(ASSETS_PATH, exist_ok=True)

# Neo Theme Colors
BG_DARK = "#0e0e0e"
FG_NEON = "#00f7ff"
BTN_BG = "#1a1a1a"
BTN_BORDER = "#00f7ff"
FONT_HEADER = ("Orbitron", 16, "bold")
FONT_BODY = ("Share Tech Mono", 12)

class NeoTracker:
    def __init__(self, root):
        self.root = root
        self.root.title("🕶️ NeoTracker")
        self.root.geometry("820x700")
        self.root.configure(bg=BG_DARK)
        self.start_time = None
        self.setup_gui()

    def neo_button(self, text, command):
        btn = tk.Button(self.root, text=text, command=command,
                        bg=BTN_BG, fg=FG_NEON,
                        activebackground=FG_NEON,
                        activeforeground=BG_DARK,
                        font=FONT_BODY,
                        relief="flat", bd=3,
                        highlightbackground=BTN_BORDER,
                        highlightcolor=BTN_BORDER,
                        highlightthickness=1,
                        padx=12, pady=6)
        return btn

    def setup_gui(self):
        tk.Label(self.root, text="🌐 NeoTracker Interface", bg=BG_DARK,
                 fg=FG_NEON, font=FONT_HEADER).pack(pady=20)

        self.neo_button("🟢 Start Session", self.start_session).pack(pady=10)
        self.neo_button("🛑 Stop Session", self.stop_session).pack(pady=10)
        self.neo_button("📊 Show Stats", self.show_stats).pack(pady=10)
        self.neo_button("📆 Generate Heatmap", self.generate_heatmap).pack(pady=10)

        self.img_label = tk.Label(self.root, bg=BG_DARK)
        self.img_label.pack(pady=25)

    def start_session(self):
        self.start_time = time.time()
        messagebox.showinfo("🟢 Tracking", "Session started... Stay focused!")

    def stop_session(self):
        if self.start_time is None:
            messagebox.showerror("⚠️ Error", "No session active. Click Start first.")
            return

        duration = int(time.time() - self.start_time)
        date_str = datetime.now().strftime("%Y-%m-%d")
        self.start_time = None

        with open(LOG_PATH, "a") as f:
            f.write(f"{date_str},{duration}\n")

        messagebox.showinfo("✅ Saved", f"Session saved: {duration} seconds")

    def show_stats(self):
        if not os.path.exists(LOG_PATH):
            messagebox.showinfo("📭 No Data", "No session data found.")
            return

        df = pd.read_csv(LOG_PATH, names=["Date", "Duration (seconds)"])
        df["Duration (seconds)"] = pd.to_numeric(df["Duration (seconds)"], errors="coerce")
        total = df["Duration (seconds)"].sum()
        average = df["Duration (seconds)"].mean()

        messagebox.showinfo("📊 Session Stats",
                            f"🧠 Total Tracked: {total:.0f} sec\n⏱️ Daily Average: {average:.1f} sec")

    def generate_heatmap(self):
        try:
            df = pd.read_csv(LOG_PATH, names=["Date", "Duration (seconds)"])
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df["Duration (seconds)"] = pd.to_numeric(df["Duration (seconds)"], errors="coerce")
            df.dropna(inplace=True)
            df["Day"] = df["Date"].dt.day
            df["Month"] = df["Date"].dt.month

            pivot = df.groupby(["Month", "Day"])["Duration (seconds)"].sum().unstack(fill_value=0)
            pivot = pivot.astype(float)

            plt.figure(figsize=(12, 4))
            sns.heatmap(pivot, cmap="rocket", linewidths=0.5, linecolor="#101010", cbar=True)
            plt.title("🔮 Productivity Glow Map", fontsize=14)
            plt.xlabel("Day")
            plt.ylabel("Month")
            plt.tight_layout()

            img_path = f"{ASSETS_PATH}/heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(img_path, dpi=100)
            plt.close()

            self.display_heatmap(img_path)

        except Exception as e:
            messagebox.showerror("💥 Heatmap Error", f"Generation failed:\n{e}")

    def display_heatmap(self, path):
        img = Image.open(path).resize((760, 300))
        img_tk = ImageTk.PhotoImage(img)
        self.img_label.config(image=img_tk)
        self.img_label.image = img_tk

# 🧠 Launch app
if __name__ == "__main__":
    root = tk.Tk()
    app = NeoTracker(root)
    root.mainloop()