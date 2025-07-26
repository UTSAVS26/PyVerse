import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
from scipy.stats import norm
import os

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load LMS reference data using safe, relative paths
lms_0_5 = pd.read_csv(os.path.join(BASE_DIR, 'data', 'children_0_5_data.csv'))      # Month, Sex, L, M, S
lms_5_19 = pd.read_csv(os.path.join(BASE_DIR, 'data', 'children_5_19_data.csv'))    # Month, Sex, L, M, S


def get_lms_0_5(age_months, sex):
    row = lms_0_5[(lms_0_5['Month'] == age_months) & (lms_0_5['Sex'] == sex)]
    if row.empty:
        return None
    return row.iloc[0]['L'], row.iloc[0]['M'], row.iloc[0]['S']

def get_lms_5_19(age_months, sex):
    ages = lms_5_19['Age'].unique()
    nearest_age = min(ages, key=lambda x: abs(x - age_months))
    row = lms_5_19[(lms_5_19['Age'] == nearest_age) & (lms_5_19['Sex'] == sex)]
    if row.empty:
        return None
    return row.iloc[0]['L'], row.iloc[0]['M'], row.iloc[0]['S']

def classify_bmi_child(z):
    if z < -2:
        return "Underweight"
    elif z < 1:
        return "Normal"
    elif z < 2:
        return "Overweight"
    else:
        return "Obese"

def classify_bmi_adult(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"

def interpret_whr(whr, sex):
    # WHO cutoffs for adults
    if sex == 'M':
        return "High risk" if whr > 0.90 else "Low risk"
    else:
        return "High risk" if whr > 0.85 else "Low risk"

def calculate_bmi():
    try:
        weight = float(weight_entry.get())
        height = float(height_entry.get())
        age = float(age_entry.get())
        waist = float(waist_entry.get())
        hip = float(hip_entry.get())

        weight_unit = weight_unit_var.get()
        height_unit = height_unit_var.get()
        age_unit = age_unit_var.get()
        waist_unit = waist_unit_var.get()
        hip_unit = hip_unit_var.get()
        sex = sex_var.get()

        # Unit conversions
        if weight_unit == 'lb':
            weight *= 0.453592
        if height_unit == 'cm':
            height /= 100
        elif height_unit == 'in':
            height *= 0.0254
        if waist_unit == 'in':
            waist *= 2.54
        if hip_unit == 'in':
            hip *= 2.54
        # Now waist and hip are in cm

        if age_unit == 'years':
            age_months = int(age * 12)
        else:
            age_months = int(age)
        age_years = age_months / 12

        bmi = weight / (height ** 2)
        bmi_result.config(text=f"BMI: {bmi:.2f}")

        # Determine age group for LMS
        if age_months <= 60:
            lms = get_lms_0_5(age_months, sex)
        elif 60 < age_months < 228:  # 5 to 19 years in months
            lms = get_lms_5_19(age_months, sex)
        else:
            category = classify_bmi_adult(bmi)
            category_result.config(text=f"Category: {category}")
            whr = waist / hip if hip > 0 else 0
            whr_result.config(text=f"WHR: {whr:.2f} ({interpret_whr(whr, sex)})")
            return

        if lms is None:
            messagebox.showerror("Error", "No LMS data for this age/sex")
            return

        L, M, S = lms
        if L == 0:
            z = np.log(bmi / M) / S
        else:
            z = ((bmi / M) ** L - 1) / (L * S)

        percentile = norm.cdf(z) * 100
        category = classify_bmi_child(z)
        category_result.config(text=f"{category} (Z: {z:.2f}, Percentile: {percentile:.1f}%)")

        # WHR calculation
        whr = waist / hip if hip > 0 else 0
        whr_result.config(text=f"WHR: {whr:.2f} ({interpret_whr(whr, sex)})")

    except ValueError:
        messagebox.showerror("Error", "Please enter valid numbers.")

# -------------------- UI --------------------

root = tk.Tk()
root.title("BMI Calculator")
root.geometry("400x420")
root.resizable(False, False)

# Weight
tk.Label(root, text="Weight:").grid(row=0, column=0, padx=10, pady=5, sticky="e")
weight_entry = tk.Entry(root)
weight_entry.grid(row=0, column=1)
weight_unit_var = tk.StringVar(value='kg')
tk.OptionMenu(root, weight_unit_var, 'kg', 'lb').grid(row=0, column=2)

# Height
tk.Label(root, text="Height:").grid(row=1, column=0, padx=10, pady=5, sticky="e")
height_entry = tk.Entry(root)
height_entry.grid(row=1, column=1)
height_unit_var = tk.StringVar(value='m')
tk.OptionMenu(root, height_unit_var, 'm', 'cm', 'in').grid(row=1, column=2)

# Age
tk.Label(root, text="Age:").grid(row=2, column=0, padx=10, pady=5, sticky="e")
age_entry = tk.Entry(root)
age_entry.grid(row=2, column=1)
age_unit_var = tk.StringVar(value='years')
tk.OptionMenu(root, age_unit_var, 'months', 'years').grid(row=2, column=2)

# Sex
tk.Label(root, text="Sex:").grid(row=3, column=0, padx=10, pady=5, sticky="e")
sex_var = tk.StringVar(value='M')
tk.OptionMenu(root, sex_var, 'M', 'F').grid(row=3, column=1)

# Waist Circumference
tk.Label(root, text="Waist Circumference:").grid(row=4, column=0, padx=10, pady=5, sticky="e")
waist_entry = tk.Entry(root)
waist_entry.grid(row=4, column=1)
waist_unit_var = tk.StringVar(value='cm')
tk.OptionMenu(root, waist_unit_var, 'cm', 'in').grid(row=4, column=2)

# Hip Circumference
tk.Label(root, text="Hip Circumference:").grid(row=5, column=0, padx=10, pady=5, sticky="e")
hip_entry = tk.Entry(root)
hip_entry.grid(row=5, column=1)
hip_unit_var = tk.StringVar(value='cm')
tk.OptionMenu(root, hip_unit_var, 'cm', 'in').grid(row=5, column=2)

# Calculate Button
tk.Button(root, text="Calculate", command=calculate_bmi).grid(row=6, column=0, columnspan=3, pady=15)

# Results
bmi_result = tk.Label(root, text="BMI: ", font=("Arial", 10, "bold"))
bmi_result.grid(row=7, column=0, columnspan=3, pady=5)

category_result = tk.Label(root, text="Category: ", font=("Arial", 10))
category_result.grid(row=8, column=0, columnspan=3, pady=5)

whr_result = tk.Label(root, text="WHR: ", font=("Arial", 10))
whr_result.grid(row=9, column=0, columnspan=3, pady=5)

root.mainloop()