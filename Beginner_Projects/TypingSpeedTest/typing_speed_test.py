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
        # Reset any previous state, as suggested in the screenshot
        self.entry.config(state=tk.NORMAL)  # Ensure entry is enabled
        self.result_label.config(text="")  # Clear previous results
        self.button_start.config(state=tk.DISABLED, text="Test in Progress...")  # Disable button during test

        self.sentence = random.choice(self.sentences)
        self.label_sentence.config(text=self.sentence)
        self.entry.delete(0, tk.END)
        self.start_time = timer()
        self.entry.focus()

    def _finish_test(self):
        # Handle case where test hasn't started
        if not self.start_time:
            return  # Test not started

        user_input = self.entry.get()
        self.entry.config(state=tk.DISABLED)
        # Re-enable the start button and reset its text
        self.button_start.config(state=tk.NORMAL, text="Start Test")

        end_time = timer()
        elapsed = end_time - self.start_time

        # Prevent division by zero and handle very short times
        elapsed = max(elapsed, 0.1)

        # Words per minute (WPM) - use the standard formula: (characters/5)/minutes
        # This accounts for varying word lengths more accurately
        char_count = len(user_input.strip())
        wpm = (char_count / 5) / (elapsed / 60) if elapsed > 0 else 0

        # More comprehensive accuracy calculation
        min_length = min(len(user_input), len(self.sentence))
        max_length = max(len(user_input), len(self.sentence))

        # Count matching characters at same positions
        correct_chars = sum(
            1 for i in range(min_length)
            if user_input[i] == self.sentence[i]
        )

        # Calculate accuracy considering total expected characters
        # Penalize for missing or extra characters
        accuracy = (correct_chars / max_length) * 100 if max_length > 0 else 0

        # Show results
        self.result_label.config(
            text=f"Time: {elapsed:.2f}s  |  WPM: {wpm:.2f}  |  Accuracy: {accuracy:.1f}%",
            fg="green"
        )

if __name__ == "__main__":
    root = tk.Tk()
    TypingSpeedTest(root)
    root.mainloop()
