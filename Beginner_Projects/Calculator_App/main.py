import math
import ast
import operator as op
import tkinter as tk
from tkinter import ttk
from math import factorial, log


dark_theme = {
    'bg': '#2c3e50',
    'button_bg': '#34495e',
    'button_fg': 'white',
    'operator_bg': '#e67e22',
    'equal_bg': '#27ae60',
    'clear_bg': '#c0392b',
    'entry_bg': '#34495e',
    'entry_fg': 'white'
}

light_theme = {
    'bg': '#f0f0f0',
    'button_bg': '#dcdcdc',
    'button_fg': 'black',
    'operator_bg': '#f39c12',
    'equal_bg': '#2ecc71',
    'clear_bg': '#e74c3c',
    'entry_bg': 'white',
    'entry_fg': 'black'
}


class ScientificCalculatorGUI:
    def __init__(self, master):
        self.master = master
        master.title("Scientific Calculator")
        master.geometry("500x600")

        self.result_var = tk.StringVar(value="0")

        self.create_style()
        self.create_widgets()
        self.create_theme_toggle()

        # Apply default theme
        self.apply_theme(dark_theme)

    def create_style(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        style.configure('TFrame', background=dark_theme['bg'])
        style.configure('TButton', 
                        font=('Arial', 14, 'bold'), 
                        background=dark_theme['button_bg'], 
                        foreground=dark_theme['button_fg'],
                        borderwidth=0,
                        focuscolor='none',
                        padding=10)
        style.map('TButton', 
                  background=[('active', '#2980b9'), ('pressed', '#3498db')])
        
        style.configure('Operator.TButton', 
                        background=dark_theme['operator_bg'],
                        font=('Arial', 16, 'bold'))
        style.map('Operator.TButton', 
                  background=[('active', '#d35400'), ('pressed', '#e67e22')])
        
        style.configure('Equal.TButton', 
                        background=dark_theme['equal_bg'],
                        font=('Arial', 16, 'bold'))
        style.map('Equal.TButton', 
                  background=[('active', '#2ecc71'), ('pressed', '#27ae60')])
        
        style.configure('Clear.TButton', 
                        background=dark_theme['clear_bg'],
                        font=('Arial', 16, 'bold'))
        style.map('Clear.TButton', 
                  background=[('active', '#e74c3c'), ('pressed', '#c0392b')])


    def create_widgets(self):
        # Result display
        result_frame = ttk.Frame(self.master, padding="10")
        result_frame.pack(fill=tk.X, pady=10)


        self.result_entry = ttk.Entry(result_frame, 
                                 textvariable=self.result_var, 
                                 font=('Arial', 24), 
                                 justify='right', 
                                 state='readonly')
        self.result_entry.pack(fill=tk.X, ipady=10)


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
        for j in range(6):
            button_frame.grid_columnconfigure(j, weight=1)


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
        expr = self.result_var.get()
        try:
            result = self.safe_eval(expr)
            self.result_var.set(f"{result:.10g}")
        except (ZeroDivisionError, ValueError, SyntaxError, OverflowError):
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


    def apply_theme(self, theme):
        style = ttk.Style()
        self.master.configure(bg=theme['bg'])

        style.configure('TFrame', background=theme['bg'])

        style.configure('TButton', 
                        background=theme['button_bg'], 
                        foreground=theme['button_fg'])
        style.map('TButton', 
                  background=[('active', '#2980b9'), ('pressed', '#3498db')])

        style.configure('Operator.TButton', background=theme['operator_bg'])
        style.map('Operator.TButton', 
                  background=[('active', '#d35400'), ('pressed', '#e67e22')])

        style.configure('Equal.TButton', background=theme['equal_bg'])
        style.map('Equal.TButton', 
                  background=[('active', '#2ecc71'), ('pressed', '#27ae60')])

        style.configure('Clear.TButton', background=theme['clear_bg'])
        style.map('Clear.TButton', 
                  background=[('active', '#e74c3c'), ('pressed', '#c0392b')])

        style.configure('TEntry', fieldbackground=theme['entry_bg'], foreground=theme['entry_fg'])

        # Apply entry widget colors
        self.result_entry.config(background=theme['entry_bg'], foreground=theme['entry_fg'])


    def create_theme_toggle(self):
        toggle_frame = ttk.Frame(self.master)
        toggle_frame.pack(pady=5)

        self.theme_var = tk.StringVar(value='dark')
        dark_radio = ttk.Radiobutton(toggle_frame, text="Dark Mode", value='dark', variable=self.theme_var, command=self.on_theme_change)
        light_radio = ttk.Radiobutton(toggle_frame, text="Light Mode", value='light', variable=self.theme_var, command=self.on_theme_change)
        dark_radio.pack(side=tk.LEFT, padx=5)
        light_radio.pack(side=tk.LEFT, padx=5)


    def on_theme_change(self):
        selected = self.theme_var.get()
        if selected == 'dark':
            self.apply_theme(dark_theme)
        else:
            self.apply_theme(light_theme)


if __name__ == "__main__":
    root = tk.Tk()
    app = ScientificCalculatorGUI(root)
    root.mainloop()
