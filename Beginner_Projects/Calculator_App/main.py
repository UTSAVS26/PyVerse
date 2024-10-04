import math
import tkinter as tk
from tkinter import ttk

class ScientificCalculatorGUI:
    def __init__(self, master):
        self.master = master
        master.title("Scientific Calculator")
        master.geometry("500x600")
        master.configure(bg='#2c3e50')

        self.result_var = tk.StringVar()
        self.result_var.set("0")

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
            ('7', '8', '9', '/', 'sin'),
            ('4', '5', '6', '*', 'cos'),
            ('1', '2', '3', '-', 'tan'),
            ('0', '.', '=', '+', 'sqrt'),
            ('(', ')', 'C', 'exp', 'log')
        ]

        # Create buttons
        for i, row in enumerate(buttons):
            for j, text in enumerate(row):
                if text in ['+', '-', '*', '/']:
                    style = 'Operator.TButton'
                elif text == '=':
                    style = 'Equal.TButton'
                elif text == 'C':
                    style = 'Clear.TButton'
                else:
                    style = 'TButton'
                
                button = ttk.Button(button_frame, 
                                    text=text, 
                                    style=style,
                                    command=lambda x=text: self.on_button_click(x))
                button.grid(row=i, column=j, sticky="nsew", padx=4, pady=4)

        # Configure grid
        for i in range(5):
            button_frame.grid_rowconfigure(i, weight=1)
        for i in range(5):
            button_frame.grid_columnconfigure(i, weight=1)

    def on_button_click(self, key):
        if key == '=':
            try:
                result = eval(self.result_var.get())
                self.result_var.set(str(result))
            except:
                self.result_var.set("Error")
        elif key == 'C':
            self.result_var.set("0")
        elif key in ['sin', 'cos', 'tan', 'sqrt', 'log', 'exp']:
            try:
                value = float(self.result_var.get())
                if key == 'sin':
                    result = math.sin(math.radians(value))
                elif key == 'cos':
                    result = math.cos(math.radians(value))
                elif key == 'tan':
                    result = math.tan(math.radians(value))
                elif key == 'sqrt':
                    result = math.sqrt(value)
                elif key == 'log':
                    result = math.log10(value)
                elif key == 'exp':
                    result = math.exp(value)
                self.result_var.set(str(result))
            except:
                self.result_var.set("Error")
        else:
            if self.result_var.get() == "0" or self.result_var.get() == "Error":
                self.result_var.set(key)
            else:
                self.result_var.set(self.result_var.get() + key)

if __name__ == "__main__":
    root = tk.Tk()
    app = ScientificCalculatorGUI(root)
    root.mainloop()