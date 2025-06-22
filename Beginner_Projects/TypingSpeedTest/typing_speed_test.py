import tkinter as tk
from timeit import default_timer as timer
import random

class TypingSpeedTest:
    def __init__(self, root):
        self.root = root
        self.root.title("Typing Speed Test")
        self.root.geometry("600x300")

        # Sample sentences for the test
        self.sentences = [
            "The quick brown fox jumps over the lazy dog.",
            "Practice makes perfect, so keep typing every day.",
            "Python is a powerful programming language."
        ]

        self.start_time = None
        self._setup_ui()

    def _setup_ui(self):
        self.label_sentence = tk.Label(self.root, text="", font=("Arial", 14), wraplength=500)
        self.label_sentence.pack(pady=10)

        self.entry = tk.Entry(self.root, width=60)
        self.entry.pack(pady=10)
        self.entry.bind("<Return>", lambda e: self._finish_test())

        self.button_start = tk.Button(self.root, text="Start Test", command=self._start_test)
        self.button_start.pack(pady=5)

        self.result_label = tk.Label(self.root, text="", font=("Arial", 12))
        self.result_label.pack(pady=10)

    def _start_test(self):
        self.sentence = random.choice(self.sentences)
        self.label_sentence.config(text=self.sentence)
        self.entry.delete(0, tk.END)
        self.entry.config(state=tk.NORMAL)
        self.result_label.config(text="")
        self.start_time = timer()
        self.entry.focus()

    def _finish_test(self):
        user_input = self.entry.get()
        self.entry.config(state=tk.DISABLED)
        end_time = timer()
        elapsed = end_time - self.start_time

        # Words per minute (WPM)
        wpm = len(user_input.split()) / (elapsed / 60) if elapsed > 0 else 0

        # Improved accuracy calculation across full length :contentReference[oaicite:2]{index=2}
        correct_chars = sum(
            1 for i, ch in enumerate(user_input)
            if i < len(self.sentence) and ch == self.sentence[i]
        )
        accuracy = (correct_chars / len(self.sentence)) * 100 if self.sentence else 0

        # Show results
        self.result_label.config(
            text=f"Time: {elapsed:.2f}s  |  WPM: {wpm:.2f}  |  Accuracy: {accuracy:.1f}%",
            fg="green"
        )

if __name__ == "__main__":
    root = tk.Tk()
    TypingSpeedTest(root)
    root.mainloop()
