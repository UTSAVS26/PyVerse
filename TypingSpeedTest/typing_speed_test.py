import tkinter as tk
import random
from timeit import default_timer as timer

class SpeedTypingTest:
    def __init__(self, root):
        self.root = root
        self.root.title("Typing Speed Test")
        self.root.geometry("600x350")

        # You can expand this list or load from an external file
        self.sentences = [
            "The quick brown fox jumps over the lazy dog.",
            "Speed typing improves accuracy and efficiency.",
            "Python is a powerful programming language."
        ]

        self.start_time = None
        self.setup_ui()

    def setup_ui(self):
        self.sentence = random.choice(self.sentences)

        self.label_sentence = tk.Label(self.root, text=self.sentence, font=("Arial", 14), wraplength=500)
        self.label_sentence.pack(pady=20)

        self.label_prompt = tk.Label(self.root, text="Type the above sentence and press Enter or click 'Done':",
                                     font=("Arial", 12))
        self.label_prompt.pack()

        self.entry = tk.Entry(self.root, width=60)
        self.entry.pack(pady=10)
        self.entry.bind("<Return>", lambda e: self.check_result())

        self.button_done = tk.Button(self.root, text="Done", command=self.check_result, width=10, bg="lightblue")
        self.button_done.pack(side=tk.LEFT, padx=20, pady=10)

        self.button_retry = tk.Button(self.root, text="Try Again", command=self.reset_test, width=10, bg="lightgrey")
        self.button_retry.pack(side=tk.RIGHT, padx=20, pady=10)

        self.result_label = tk.Label(self.root, text="", font=("Arial", 12))
        self.result_label.pack(pady=20)

        self.start_time = timer()

    def check_result(self):
        typed = self.entry.get()
        if typed == self.sentence:
            end_time = timer()
            time_taken = round(end_time - self.start_time, 2)
            words = len(self.sentence.split())
            wpm = round((words / time_taken) * 60, 2)
            self.result_label.config(text=f"Time: {time_taken}s WPM: {wpm}", fg="green")
        else:
            self.result_label.config(text="Text mismatch, please try again.", fg="red")

    def reset_test(self):
        self.sentence = random.choice(self.sentences)
        self.label_sentence.config(text=self.sentence)
        self.entry.delete(0, tk.END)
        self.result_label.config(text="")
        self.start_time = timer()

if __name__ == "__main__":
    root = tk.Tk()
    app = SpeedTypingTest(root)
    root.mainloop()
