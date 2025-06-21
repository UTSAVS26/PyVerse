import tkinter as tk
from timeit import default_timer as timer
import random

SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Python programming is fun and powerful.",
    "Practice typing daily to improve your speed."
]
TEST_DURATION = 60  # in seconds

class TypingSpeedTest:
    def __init__(self, root):
        self.root = root
        self.root.title("Typing Speed Test")
        self.root.geometry("700x400")
        self.start_time = None
        self.reset_vars()

        # UI setup
        self.label = tk.Label(root, text="Click 'Start' to begin", font=("Arial", 14), wraplength=600)
        self.label.pack(pady=20)

        self.entry = tk.Entry(root, font=("Arial", 12), width=80, state=tk.DISABLED)
        self.entry.pack(pady=10)
        self.entry.bind("<Return>", self.calculate_result)

        self.timer_label = tk.Label(root, text="", font=("Arial", 12))
        self.timer_label.pack()

        self.result_label = tk.Label(root, text="", font=("Arial", 14))
        self.result_label.pack(pady=20)

        self.start_button = tk.Button(root, text="Start", command=self.start_test)
        self.start_button.pack(side=tk.LEFT, padx=10)

        self.reset_button = tk.Button(root, text="Reset", command=self.reset_test, state=tk.DISABLED)
        self.reset_button.pack(side=tk.LEFT)

    def reset_vars(self):
        self.target = ""
        self.time_elapsed = 0

    def start_test(self):
        self.reset_test()
        self.target = random.choice(SENTENCES)
        self.label.config(text=self.target)
        self.start_time = timer()
        self.entry.config(state=tk.NORMAL)
        self.entry.focus()
        self.start_button.config(state=tk.DISABLED)
        self.reset_button.config(state=tk.NORMAL)
        self.update_timer()

    def update_timer(self):
        if self.start_time is None:
            return

        elapsed = timer() - self.start_time
        remaining = max(0, TEST_DURATION - int(elapsed))
        self.timer_label.config(text=f"Time left: {remaining}s")
        if remaining > 0:
            self.root.after(1000, self.update_timer)
        else:
            self.calculate_result()

    def calculate_result(self, event=None):
        if self.start_time is None:
            return

        input_text = self.entry.get()
        elapsed = timer() - self.start_time
        minutes = elapsed / 60 if elapsed > 0 else 1
        words_typed = len(input_text.split())
        wpm = words_typed / minutes

        correct_chars = sum(1 for i, c in enumerate(input_text) if i < len(self.target) and c == self.target[i])
        accuracy = (correct_chars / len(self.target)) * 100 if self.target else 0

        self.result_label.config(
            text=f"WPM: {wpm:.2f} | Accuracy: {accuracy:.2f}%"
        )

        self.entry.config(state=tk.DISABLED)
        self.start_time = None

    def reset_test(self):
        self.reset_vars()
        self.label.config(text="Click 'Start' to begin")
        self.entry.config(state=tk.DISABLED)
        self.entry.delete(0, tk.END)
        self.timer_label.config(text="")
        self.result_label.config(text="")
        self.start_button.config(state=tk.NORMAL)

if __name__ == "__main__":
    root = tk.Tk()
    TypingSpeedTest(root)
    root.mainloop()
