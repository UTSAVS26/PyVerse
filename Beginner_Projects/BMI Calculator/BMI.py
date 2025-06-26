import tkinter as tk
from tkinter import messagebox

def calculate_bmi():
    try:
        weight = float(weight_entry.get())
        height = float(height_entry.get())

        if weight <= 0 or height <= 0:
            raise ValueError

        bmi = round(weight / (height ** 2), 2)
        bmi_result.config(text=f"BMI: {bmi}")

        if bmi < 18.5:
            category = "Underweight"
        elif bmi < 24.9:
            category = "Normal weight"
        elif bmi < 29.9:
            category = "Overweight"
        else:
            category = "Obesity"

        category_result.config(text=f"Category: {category}")
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid positive numbers for weight and height.")

root = tk.Tk()
root.title("BMI Calculator")
root.geometry("300x220")
root.resizable(False, False)

# Weight input
tk.Label(root, text="Enter weight (kg):").grid(row=0, column=0, padx=10, pady=10, sticky="e")
weight_entry = tk.Entry(root)
weight_entry.grid(row=0, column=1, padx=10)

# Height input
tk.Label(root, text="Enter height (m):").grid(row=1, column=0, padx=10, pady=10, sticky="e")
height_entry = tk.Entry(root)
height_entry.grid(row=1, column=1, padx=10)

# Calculate button
tk.Button(root, text="Calculate BMI", command=calculate_bmi).grid(row=2, column=0, columnspan=2, pady=15)

# Result labels
bmi_result = tk.Label(root, text="BMI: ", font=("Arial", 10, "bold"))
bmi_result.grid(row=3, column=0, columnspan=2, pady=5)

category_result = tk.Label(root, text="Category: ", font=("Arial", 10))
category_result.grid(row=4, column=0, columnspan=2)

root.mainloop()
