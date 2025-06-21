import tkinter as tk
from timeit import default_timer as timer
import random

class TypingSpeedTest:
    def __init__(self, root):
        self.root = root
        self.root.title("Typing Speed Test")
        self.sentences = [
            "Python is a versatile programming language.",
            "Typing tests are great for improving speed.",
            "Practice makes perfect when learning to code.",
            "The quick brown fox jumps over the lazy dog."
        ]
        self.start_time = None

        self.label_instruction = tk.Label(root, text="Click Start to begin the test", font=("Helvetica", 14))
        self.label_instruction.pack(pady=10)

        self.label_sentence = tk.Label(root, text="", wraplength=500, font=("Helvetica", 12))
        self.label_sentence.pack(pady=10)

        self.entry = tk.Entry(root, width=60, font=("Helvetica", 12))
        self.entry.pack(pady=10)

        self.result_label = tk.Label(root, text="", font=("Helvetica", 12))
        self.result_label.pack(pady=10)

        self.start_button = tk.Button(root, text="Start", command=self._start_test)
        self.start_button.pack(pady=5)

        self.done_button = tk.Button(root, text="Done", command=self._finish_test)
        self.done_button.pack(pady=5)

    def _start_test(self):
        # Reset any previous test state
        if hasattr(self, 'sentence'):
            self._reset_test()

        self.sentence = random.choice(self.sentences)
        self.label_sentence.config(text=self.sentence)
        self.entry.delete(0, tk.END)
        self.entry.config(state=tk.NORMAL)
        self.result_label.config(text="")
        self.start_time = timer()
        self.entry.focus()

    def _finish_test(self):
        if self.start_time is None:
            return  # No test in progress

        user_input = self.entry.get()
        if not user_input.strip():
            self.result_label.config(text="Please type something before finishing!", fg="red")
            return

        self.entry.config(state=tk.DISABLED)
        end_time = timer()
        elapsed = end_time - self.start_time

        correct_chars = 0
        max_length = max(len(user_input), len(self.sentence))
        for i in range(max_length):
            if i < len(user_input) and i < len(self.sentence) and user_input[i] == self.sentence[i]:
                correct_chars += 1

        accuracy = (correct_chars / len(self.sentence)) * 100 if len(self.sentence) > 0 else 0
        time_in_minutes = elapsed / 60
        words_typed = len(user_input.split())
        wpm = words_typed / time_in_minutes if time_in_minutes > 0 else 0

        result = f"WPM: {wpm:.2f}\nAccuracy: {accuracy:.2f}%\nTime Elapsed: {elapsed:.2f} seconds"
        self.result_label.config(text=result, fg="green")

    def _reset_test(self):
        """Reset the test to initial state"""
        self.entry.config(state=tk.NORMAL)
        self.entry.delete(0, tk.END)
        self.label_sentence.config(text="")
        self.result_label.config(text="")
        self.start_time = None

if __name__ == "__main__":
    root = tk.Tk()
    app = TypingSpeedTest(root)
    root.mainloop()
