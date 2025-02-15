import math
import tkinter as tk
from tkinter import ttk
from math import factorial, log

class ScientificCalculatorGUI:
    def __init__(self, master):
        self.master = master
        master.title("Scientific Calculator")
        master.geometry("500x600")
        master.configure(bg='#2c3e50')

        self.result_var = tk.StringVar(value="0")

        self.create_style()
        self.create_widgets()

    def create_style(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        style.configure('TFrame', background='#2c3e50')
        style.configure('TButton', 
                        font=('Arial', 14, 'bold'), 
                        background='#34495e', 
                        foreground='white',
                        borderwidth=0,
                        focuscolor='none',
                        padding=10)
        style.map('TButton', 
                  background=[('active', '#2980b9'), ('pressed', '#3498db')])
        
        style.configure('Operator.TButton', 
                        background='#e67e22',
                        font=('Arial', 16, 'bold'))
        style.map('Operator.TButton', 
                  background=[('active', '#d35400'), ('pressed', '#e67e22')])
        
        style.configure('Equal.TButton', 
                        background='#27ae60',
                        font=('Arial', 16, 'bold'))
        style.map('Equal.TButton', 
                  background=[('active', '#2ecc71'), ('pressed', '#27ae60')])
        
        style.configure('Clear.TButton', 
                        background='#c0392b',
                        font=('Arial', 16, 'bold'))
        style.map('Clear.TButton', 
                  background=[('active', '#e74c3c'), ('pressed', '#c0392b')])

    def create_widgets(self):
        # Result display
        result_frame = ttk.Frame(self.master, padding="10")
        result_frame.pack(fill=tk.X, pady=10)

        result_entry = ttk.Entry(result_frame, 
                                 textvariable=self.result_var, 
                                 font=('Arial', 24), 
                                 justify='right', 
                                 state='readonly')
        result_entry.pack(fill=tk.X, ipady=10)

        # Button frame
        button_frame = ttk.Frame(self.master, padding="10")
        button_frame.pack(fill=tk.BOTH, expand=True)

        # Define button layout
        buttons = [
            ('7', '8', '9', '/', 'sin', 'sinr'),
            ('4', '5', '6', '*', 'cos', 'cosr'),
            ('1', '2', '3', '-', 'tan', 'tanr'),
            ('0', '.', '=', '+', 'sqrt', 'logn'),
            ('(', ')', 'C', 'exp', 'log', 'fact'),
            ('pow', 'mod', ',', ' ')
        ]

        # Create buttons
        for i, row in enumerate(buttons):
            for j, text in enumerate(row):
                style = self.get_button_style(text)
                button = ttk.Button(button_frame, 
                                    text=text, 
                                    style=style,
                                    command=lambda x=text: self.on_button_click(x))
                button.grid(row=i, column=j, sticky="nsew", padx=4, pady=4)

        # Configure grid
        for i in range(6):
            button_frame.grid_rowconfigure(i, weight=1)
            button_frame.grid_columnconfigure(i, weight=1)

    def get_button_style(self, text):
        if text in ['+', '-', '*', '/', 'pow', 'mod']:
            return 'Operator.TButton'
        elif text == '=':
            return 'Equal.TButton'
        elif text == 'C':
            return 'Clear.TButton'
        return 'TButton'

    def on_button_click(self, key):
        if key == '=':
            self.calculate_result()
        elif key == 'C':
            self.result_var.set("0")
        elif key in ['sin', 'cos', 'tan', 'sqrt', 'log', 'exp', 'sinr', 'cosr', 'tanr', 'fact', 'logn', 'pow', 'mod']:
            self.calculate_function(key)
        else:
            self.update_display(key)

    def calculate_result(self):
        try:
            result = eval(self.result_var.get())
            self.result_var.set(f"{result:.10g}")
        except:
            self.result_var.set("Error")

    def calculate_function(self, func):
        try:
            if func in ['pow', 'logn', 'mod']:
                parts = self.result_var.get().split(',')
                if len(parts) != 2:
                    raise ValueError("Invalid input format")
                value = float(parts[0])
                second_value = float(parts[1])
                if func == 'pow':
                    result = math.pow(value, second_value)
                elif func == 'logn':
                    result = log(value, second_value)
                elif func == 'mod':
                    result = value % second_value
            else:
                value = float(self.result_var.get())
                if func == 'sinr':
                    result = math.sin(value)
                elif func == 'cosr':
                    result = math.cos(value)
                elif func == 'tanr':
                    result = math.tan(value)
                elif func == 'fact':
                    result = factorial(int(value))
                else:
                    result = {
                        'sin': math.sin(math.radians(value)),
                        'cos': math.cos(math.radians(value)),
                        'tan': math.tan(math.radians(value)),
                        'sqrt': math.sqrt(value),
                        'log': math.log10(value),
                        'exp': math.exp(value)
                    }[func]
            self.result_var.set(f"{result:.10g}")
        except:
            self.result_var.set("Error")

    def update_display(self, key):
        if self.result_var.get() in ["0", "Error"]:
            self.result_var.set(key)
        else:
            self.result_var.set(self.result_var.get() + key)

if __name__ == "__main__":
    root = tk.Tk()
    app = ScientificCalculatorGUI(root)
    root.mainloop()
