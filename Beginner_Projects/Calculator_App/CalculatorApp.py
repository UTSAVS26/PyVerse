import tkinter as tk
from tkinter import ttk
import math

class CalculatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Modern Calculator")
        self.root.geometry("400x500")
        self.root.configure(bg='#f0f0f0')
        self.create_widgets()

    def create_widgets(self):
        # Display
        self.display = ttk.Entry(self.root, font=('Arial', 18), justify='right')
        self.display.grid(row=0, column=0, columnspan=4, ipadx=8, ipady=20, padx=10, pady=10, sticky="nsew")

        # Number and operation buttons
        buttons = [
            ('7', 1, 0), ('8', 1, 1), ('9', 1, 2), ('/', 1, 3),
            ('4', 2, 0), ('5', 2, 1), ('6', 2, 2), ('*', 2, 3),
            ('1', 3, 0), ('2', 3, 1), ('3', 3, 2), ('-', 3, 3),
            ('0', 4, 0), ('.', 4, 1), ('+', 4, 2), ('=', 4, 3),
            ('C', 5, 0), ('sqrt', 5, 1), ('log', 5, 2), ('exp', 5, 3),
            ('sin', 6, 0), ('cos', 6, 1), ('tan', 6, 2)
        ]

        # Create buttons dynamically
        for (text, row, col) in buttons:
            ttk.Button(self.root, text=text, command=lambda t=text: self.on_button_click(t)).grid(row=row, column=col, ipadx=10, ipady=10, padx=5, pady=5, sticky="nsew")

        # Configure the rows and columns for responsive design
        for i in range(7):
            self.root.grid_rowconfigure(i, weight=1)
        for i in range(4):
            self.root.grid_columnconfigure(i, weight=1)

    def on_button_click(self, char):
        if char == 'C':
            self.display.delete(0, tk.END)
        elif char == '=':
            try:
                expression = self.display.get()
                result = eval(expression)
                self.display.delete(0, tk.END)
                self.display.insert(tk.END, str(result))
            except Exception as e:
                self.display.delete(0, tk.END)
                self.display.insert(tk.END, 'Error')
        elif char == 'sqrt':
            try:
                value = float(self.display.get())
                result = math.sqrt(value)
                self.display.delete(0, tk.END)
                self.display.insert(tk.END, str(result))
            except:
                self.display.delete(0, tk.END)
                self.display.insert(tk.END, 'Error')
        elif char == 'log':
            try:
                value = float(self.display.get())
                result = math.log(value)
                self.display.delete(0, tk.END)
                self.display.insert(tk.END, str(result))
            except:
                self.display.delete(0, tk.END)
                self.display.insert(tk.END, 'Error')
        elif char == 'exp':
            try:
                value = float(self.display.get())
                result = math.exp(value)
                self.display.delete(0, tk.END)
                self.display.insert(tk.END, str(result))
            except:
                self.display.delete(0, tk.END)
                self.display.insert(tk.END, 'Error')
        elif char in ['sin', 'cos', 'tan']:
            try:
                value = float(self.display.get())
                if char == 'sin':
                    result = math.sin(math.radians(value))
                elif char == 'cos':
                    result = math.cos(math.radians(value))
                elif char == 'tan':
                    result = math.tan(math.radians(value))
                self.display.delete(0, tk.END)
                self.display.insert(tk.END, str(result))
            except:
                self.display.delete(0, tk.END)
                self.display.insert(tk.END, 'Error')
        else:
            self.display.insert(tk.END, char)


# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = CalculatorApp(root)
    root.mainloop()
