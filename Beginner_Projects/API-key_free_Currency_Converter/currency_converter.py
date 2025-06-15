import tkinter as tk
import requests # type: ignore
from tkinter import messagebox

#keeping the intial theme as light
is_dark_mode = False

#the different themes:
Themes = {
    "light":{
        "background_color": "#E0F7FA",
        "foreground_color": "#333333",
        "entry_bg": "#F0F0F0",
        "button_background": "#4CAF50",
        "button_forrground": "#F0F0F0"
    },
    "dark":{
        "background_color": "#2C2C2C",
        "foreground_color": "#EEEEEE",
        "entry_bg": "#292929",
        "button_background": "#FFB300",
        "button_forrground": "#F0F7FA"
    }
}

#window components setting
root = tk.Tk()
root.title("Currency Converter")
root.geometry("360x380")
root.resizable(False, False)

FONT_LABEL = ("Helvetica", 10, "bold")
FONT_ENTRY = ("Helvetica", 11)
FONT_RESULT = ("Helvetica", 12, "bold")

#func to apply theme
def apply_theme(theme_name):
    theme = Themes[theme_name]
    root.configure(bg=theme["background_color"])
    for widget in root.winfo_children():
        cls = widget.__class__.__name__
        if cls == "Entry":
            widget.configure(bg=theme["entry_bg"], fg=theme["foreground_color"])
        elif cls == "Button":
            if widget['text'] == "Switch Theme":
                widget.configure(bg="#888", fg="#F0F0F0")
            else:
                widget.configure(bg=theme["button_background"], fg=theme["button_forrground"])
        elif cls == "Label":
            widget.configure(bg=theme["background_color"], fg=theme["foreground_color"])

#toggle func
def toggle_theme():
    global is_dark_mode
    is_dark_mode = not is_dark_mode
    theme = "dark" if is_dark_mode else "light"
    apply_theme(theme)

#conversion logic
def convert_currency():
    #definig them
    from_curr = entry_from.get().upper().strip()
    to_curr = entry_to.get().upper().strip()
    amount = entry_amount.get().strip()

    if from_curr in ["", "eg: USD"] or to_curr in ["", "eg: INR"] or amount in ["", "eg: 250.00"]:
        error_label.config(text="Please fill all fields with valid values.")
        return
    
    try:
        amount = float(amount)
    except ValueError:
        error_label.config(text="Invalid amount. Please enter a valid number.")
        return
    
    #api url
    url = f"https://api.frankfurter.app/latest?amount={amount}&from={from_curr}&to={to_curr}"

    try:
        response = requests.get(url)
        data = response.json()

        print(data)

        converted = data["rates"].get(to_curr)
        if converted:
            result_label.config(text=f"{amount} {from_curr} = {round(converted, 2)} {to_curr}")
            error_label.config(text="") 
        else:
            error_label.config(text="Conversion failed. Check currency codes.")
    except Exception as e:
        error_label.config(text="An error occurred. Check your internet connection and try again.")


#label+entry func
def labeled_entry(labe_text, placeholder):
    label = tk.Label(root, text=labe_text, font=FONT_LABEL)
    label.pack(pady=(10,2))
    entry = tk.Entry(root, font=FONT_ENTRY, width=30, bd=2,relief="groove",fg='gray')
    entry.insert(0, placeholder)

    def on_focus_in(event):
        if entry.get() == placeholder:
            entry.delete(0, tk.END)
            entry.config(fg='black')
    
    def on_focus_out(event):
        if not entry.get():
            entry.insert(0, placeholder)
            entry.config(fg='gray')

    entry.bind("<FocusIn>", on_focus_in)
    entry.bind("<FocusOut>", on_focus_out)
    entry.pack()
    return entry

#Input fields:
entry_from = labeled_entry("From Currency Code (3 letters):", "eg: USD")
entry_to = labeled_entry("To Currency Code (3 letters):", "eg: INR")
entry_amount = labeled_entry("Amount:", "e.g. 250.00")

#convrtion button
convert_button = tk.Button(root, text="Convert", command=convert_currency, font=FONT_LABEL, width=15)
convert_button.pack(pady=15)

#result label:
result_label = tk.Label(root, text="", font=FONT_RESULT)
result_label.pack(pady=10)

#error label:
error_label = tk.Label(root, text="", fg="red", font=("Helvetica", 10, "italic"))
error_label.pack(pady=(5, 0))

#toggling button
toggle_button = tk.Button(root, text="Toggle Theme", command=toggle_theme, font=("Helvetica", 9, "bold"))
toggle_button.pack(pady=5)

apply_theme("light")

root.mainloop()
