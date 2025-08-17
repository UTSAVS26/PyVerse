import os
import math
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk

# Optional dependencies (per README)
try:
    import pandas as pd
except Exception as e:
    pd = None
try:
    from scipy.stats import norm
except Exception:
    # Fallback normal CDF if scipy unavailable
    def _erf(x):
        # Abramowitz & Stegun erf approximation
        # Not perfect, but fine for display purposes if scipy isn't installed
        # https://stackoverflow.com/a/457805/3219667
        sign = 1 if x >= 0 else -1
        x = abs(x)
        a1,a2,a3,a4,a5,p = 0.254829592,-0.284496736,1.421413741,-1.453152027,1.061405429,0.3275911
        t = 1.0/(1.0+p*x)
        y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*math.exp(-x*x)
        return sign*y

    class _Norm:
        @staticmethod
        def cdf(x):
            return 0.5*(1+_erf(x/math.sqrt(2)))
    norm = _Norm()


# ---------- CSV LOADING ----------
def _find_data_path(filename):
    """Try a few sensible locations for data files."""
    here = os.path.dirname(__file__)
    candidates = [
        os.path.join(here, "BMI_Calculator", "data", filename),
        os.path.join(here, "data", filename),
        os.path.join(here, filename),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def _normalize_columns(df):
    """Lowercase, strip, and simplify column names for robust access."""
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.replace(r"[:]", "", regex=True)
        .str.replace(r"SD", "SD", regex=False)  # keep SD
        .str.replace(r"\s+", "", regex=True)
        .str.lower()
    )
    return df


def load_who_data():
    """Load WHO LMS datasets; return (df_0_5, df_5_19)."""
    if pd is None:
        return None, None

    path_0_5 = _find_data_path("children_0_5_data.csv")
    path_5_19 = _find_data_path("children_5_19_data.csv")

    df_0_5 = pd.read_csv(path_0_5) if path_0_5 else None
    df_5_19 = pd.read_csv(path_5_19) if path_5_19 else None

    if df_0_5 is not None:
        df_0_5 = _normalize_columns(df_0_5)
        # Expect columns like: month, l, m, s, sex
        # Coerce numeric
        for col in ("month", "l", "m", "s"):
            if col in df_0_5.columns:
                df_0_5[col] = pd.to_numeric(df_0_5[col], errors="coerce")
        # Normalize sex values
        if "sex" in df_0_5.columns:
            df_0_5["sex"] = df_0_5["sex"].astype(str).str.strip().str.lower()

    if df_5_19 is not None:
        df_5_19 = _normalize_columns(df_5_19)
        # Some files include both "yearmonth" and "month". We'll prefer "month" if present.
        # Coerce numeric L,M,S and month if available.
        for col in ("month", "l", "m", "s"):
            if col in df_5_19.columns:
                df_5_19[col] = pd.to_numeric(df_5_19[col], errors="coerce")
        if "sex" in df_5_19.columns:
            df_5_19["sex"] = df_5_19["sex"].astype(str).str.strip().str.lower()

        # If month is missing but there's an "age" or "yearmonth" column, try to build a month index
        if "month" not in df_5_19.columns:
            # Some datasets have separate "year" and "month" fields, but you've provided a combined header.
            # If not present, we can't build a lookup by month reliably—UI will warn in that case.
            pass

    return df_0_5, df_5_19


DF_0_5, DF_5_19 = load_who_data()


# ---------- WHO HELPERS ----------
def sex_key_from_ui(val: str) -> str:
    """Map UI gender to dataset sex key."""
    if not val:
        return ""
    v = val.strip().lower()
    if v in ("male", "m"):
        return "m"
    if v in ("female", "f"):
        return "f"
    return v  # 'other' will be handled upstream
 


def lms_zscore(bmi, L, M, S):
    """Compute WHO LMS z-score for BMI."""
    if any(v is None for v in (bmi, L, M, S)):
        return None
    try:
        if L == 0:
            return math.log(bmi / M) / S
        return ((bmi / M) ** L - 1) / (L * S)
    except Exception:
        return None


def nearest_row_by_month(df, month_value, sex):
    """Pick the nearest month row for a given sex."""
    if df is None:
        return None
    # Some datasets may call it "month" or "year:month"
    month_col = "month" if "month" in df.columns else None
    if not month_col or "sex" not in df.columns:
        return None

    dd = df[df["sex"] == sex]
    if dd.empty:
        return None

    dd = dd.copy()
    dd["diff"] = (dd[month_col] - month_value).abs()
    dd = dd.sort_values("diff")
    return dd.iloc[0].to_dict()



def child_category_from_z(z):
    """WHO child BMI categories from z-score."""
    if z is None:
        return "Unknown"
    if z < -2:
        return "Underweight"
    if -2 <= z <= 1:
        return "Normal"
    if 1 < z <= 2:
        return "Overweight"
    return "Obese"  # z > 2


def category_color(name):
    return {
        "Underweight": "blue",
        "Normal": "green",
        "Overweight": "orange",
        "Obese": "red",
        "Unknown": "black",
    }.get(name, "black")


# ---------- TKINTER UI + LOGIC ----------
root = tk.Tk()
root.title("BMI Calculator (WHO-aware)")
root.geometry("380x360")
root.resizable(False, False)

# Grid config
for i in range(2):
    root.grid_columnconfigure(i, weight=1)

# Inputs
tk.Label(root, text="Enter weight (kg):").grid(row=0, column=0, padx=10, pady=8, sticky="e")
weight_entry = tk.Entry(root)
weight_entry.grid(row=0, column=1, padx=10, sticky="w")

tk.Label(root, text="Enter height (cm):").grid(row=1, column=0, padx=10, pady=8, sticky="e")
height_entry = tk.Entry(root)
height_entry.grid(row=1, column=1, padx=10, sticky="w")

tk.Label(root, text="Enter age (years):").grid(row=2, column=0, padx=10, pady=8, sticky="e")
age_entry = tk.Entry(root)
age_entry.grid(row=2, column=1, padx=10, sticky="w")

tk.Label(root, text="Select gender:").grid(row=3, column=0, padx=10, pady=8, sticky="e")
gender_var = tk.StringVar(value="Male")
gender_menu = ttk.Combobox(root, textvariable=gender_var, values=["Male", "Female", "Other"], state="readonly", width=17)
gender_menu.grid(row=3, column=1, padx=10, sticky="w")

# Buttons
def calculate_bmi():
    # Basic validations
    try:
        weight = float(weight_entry.get())
        height_cm = float(height_entry.get())
        age_years = float(age_entry.get())
        if weight <= 0 or height_cm <= 0 or age_years < 0:
            raise ValueError
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid positive numbers for weight, height, and age.")
        return

    gender = gender_var.get().strip()
    height_m = height_cm / 100.0
    bmi = round(weight / (height_m ** 2), 2)
    bmi_result.config(text=f"BMI: {bmi}")

    # Decide adult vs child logic
    if age_years >= 19:
        # Adult classification
        if bmi < 18.5:
            category = "Underweight"
        elif bmi < 25:
            category = "Normal"
        elif bmi < 30:
            category = "Overweight"
        else:
            category = "Obese"

        category_result.config(text=f"Category: {category}", fg=category_color(category))
        z_result.config(text="Z-score: —")
        pct_result.config(text="Percentile: —")
        child_note.config(text="(Adult classification used)", fg="#777")
        return

    # Child path (0–19): need Male/Female + WHO data
    sex_key = sex_key_from_ui(gender)
    if sex_key not in ("m", "f"):
        messagebox.showwarning("Missing Sex", "For children, please select Male or Female to use WHO child standards.")
        return

    if pd is None or DF_0_5 is None and DF_5_19 is None:
        messagebox.showwarning("WHO Data Missing", "WHO CSV data (or pandas) is unavailable. Cannot compute child Z-score.")
        return

    # Convert age (years) to months
    age_months = int(round(age_years * 12))

    # Choose dataset
    if age_months <= 60:
        row = nearest_row_by_month(DF_0_5, age_months, sex_key)
    else:
        row = nearest_row_by_month(DF_5_19, age_months, sex_key)

    if row is None or not all(k in row for k in ("l", "m", "s")):
        messagebox.showwarning("WHO Data Unavailable", "Could not find appropriate LMS row in WHO dataset for this age/sex.")
        return

    L, M, S = float(row["l"]), float(row["m"]), float(row["s"])
    z = lms_zscore(bmi, L, M, S)
    category = child_category_from_z(z)
    percentile = norm.cdf(z) * 100 if z is not None else None

    # Update UI
    category_result.config(text=f"Category: {category}", fg=category_color(category))
    z_result.config(text=f"Z-score: {z:.2f}" if z is not None else "Z-score: —")
    pct_result.config(text=f"Percentile: {percentile:.1f}%" if percentile is not None else "Percentile: —")
    child_note.config(text="(WHO child classification used)", fg="#777")


tk.Button(root, text="Calculate BMI", command=calculate_bmi).grid(row=4, column=0, columnspan=2, pady=14)

# Results
bmi_result = tk.Label(root, text="BMI: —", font=("Arial", 10, "bold"))
bmi_result.grid(row=5, column=0, columnspan=2, pady=(6, 2))

category_result = tk.Label(root, text="Category: —", font=("Arial", 10))
category_result.grid(row=6, column=0, columnspan=2)

z_result = tk.Label(root, text="Z-score: —", font=("Arial", 10))
z_result.grid(row=7, column=0, columnspan=2)

pct_result = tk.Label(root, text="Percentile: —", font=("Arial", 10))
pct_result.grid(row=8, column=0, columnspan=2)

child_note = tk.Label(root, text="", font=("Arial", 9, "italic"))
child_note.grid(row=9, column=0, columnspan=2, pady=(2, 0))

root.mainloop()
