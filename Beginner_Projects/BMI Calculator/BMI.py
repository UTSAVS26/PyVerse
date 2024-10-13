import tkinter as tk
from tkinter import messagebox


# Function to calculate BMI
def calculate_bmi():
    try:
        weight = float(weight_entry.get())
        height = float(height_entry.get())
        if height <= 0 or weight <= 0:
            messagebox.showerror("Input Error", "Height and weight must be positive numbers!")
            return

        bmi = weight / (height ** 2)
        bmi_result.config(text=f"BMI: {bmi:.2f}")

        # Display BMI category
        if bmi < 18.5:
            category_result.config(text="Category: Underweight")
        elif 18.5 <= bmi < 24.9:
            category_result.config(text="Category: Normal weight")
        elif 25 <= bmi < 29.9:
            category_result.config(text="Category: Overweight")
        else:
            category_result.config(text="Category: Obesity")
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numbers!")


# Create the main window
root = tk.Tk()
root.title("BMI Calculator")

# Labels and entries for weight and height
weight_label = tk.Label(root, text="Enter weight (kg):")
weight_label.grid(row=0, column=0, padx=10, pady=10)

weight_entry = tk.Entry(root)
weight_entry.grid(row=0, column=1, padx=10, pady=10)

height_label = tk.Label(root, text="Enter height (m):")
height_label.grid(row=1, column=0, padx=10, pady=10)

height_entry = tk.Entry(root)
height_entry.grid(row=1, column=1, padx=10, pady=10)

# Button to calculate BMI
calculate_button = tk.Button(root, text="Calculate BMI", command=calculate_bmi)
calculate_button.grid(row=2, column=0, columnspan=2, pady=10)

# Labels to display results
bmi_result = tk.Label(root, text="BMI: ")
bmi_result.grid(row=3, column=0, columnspan=2, pady=10)

category_result = tk.Label(root, text="Category: ")
category_result.grid(row=4, column=0, columnspan=2, pady=10)

# Start the main event loop
root.mainloop()
