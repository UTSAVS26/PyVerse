import tkinter as tk
from tkinter import messagebox

def calculate_bmi():
    try:
        weight = float(weight_entry.get())
        height_cm = float(height_entry.get())
        height = height_cm / 100  # convert to meters

        bmi = round(weight / (height ** 2), 2)
        bmi_result.config(text=f"BMI: {bmi}")

        if bmi < 18.5:
            category = "Underweight"
            color = "blue"
        elif bmi < 24.9:
            category = "Normal weight"
            color = "green"
        elif bmi < 29.9:
            category = "Overweight"
            color = "orange"
        else:
            category = "Obesity"
            color = "red"

        category_result.config(text=f"Category: {category}", fg=color)

    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid positive numbers for weight and height.")

root = tk.Tk()
root.title("BMI Calculator")
root.geometry("300x300")
root.resizable(False, False)

# Weight input
tk.Label(root, text="Enter weight (kg):").grid(row=0, column=0, padx=10, pady=10, sticky="e")
weight_entry = tk.Entry(root)
weight_entry.grid(row=0, column=1, padx=10)

# Height input
tk.Label(root, text="Enter height (cm):").grid(row=1, column=0, padx=10, pady=10, sticky="e")
height_entry = tk.Entry(root)
height_entry.grid(row=1, column=1, padx=10)

# Age input
tk.Label(root, text="Enter age:").grid(row=2, column=0, padx=10, pady=10, sticky="e")
age_entry = tk.Entry(root)
age_entry.grid(row=2, column=1, padx=10)

# Gender input
tk.Label(root, text="Select gender:").grid(row=3, column=0, padx=10, pady=10, sticky="e")
gender_var = tk.StringVar(value="Select")
gender_menu = tk.OptionMenu(root, gender_var, "Male", "Female", "Other")
gender_menu.grid(row=3, column=1, padx=10)

# Calculate button
tk.Button(root, text="Calculate BMI", command=calculate_bmi).grid(row=4, column=0, columnspan=2, pady=15)

# Result labels
bmi_result = tk.Label(root, text="BMI: ", font=("Arial", 10, "bold"))
bmi_result.grid(row=5, column=0, columnspan=2, pady=5)

category_result = tk.Label(root, text="Category: ", font=("Arial", 10))
category_result.grid(row=6, column=0, columnspan=2)

root.mainloop()
