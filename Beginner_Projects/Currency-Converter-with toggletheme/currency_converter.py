import tkinter as tk
import requests # type: ignore
from tkinter import messagebox
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY =os.getenv("API_KEY")


#Adding placeholders to avoid hardcoding
PLACEHOLDER_FROM = "eg: USD"
PLACEHOLDER_TO ="eg: INR"
PLACEHOLDER_AMOUNT = "eg: 250.00"

#keeping the intial theme as light
is_dark_mode = False

#the different themes:
Themes = {
    "light":{
        "background_color": "#E0F7FA",
        "foreground_color": "#333333",
        "entry_bg": "#F0F0F0",
        "button_background": "#4CAF50",
        "button_foreground": "#F0F0F0" #fixed typo
    },
    "dark":{
        "background_color": "#2C2C2C",
        "foreground_color": "#EEEEEE",
        "entry_bg": "#292929",
        "button_background": "#FFB300",
        "button_foreground": "#F0F7FA" #fixed typo
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
            if widget['text'] == "Toggle Theme":
                widget.configure(bg="#888", fg="#F0F0F0")
            else:
                widget.configure(bg=theme["button_background"], fg=theme["button_foreground"])
        elif cls == "Label":
            widget.configure(bg=theme["background_color"], fg=theme["foreground_color"])

#toggle func
def toggle_theme():
    global is_dark_mode
    is_dark_mode = not is_dark_mode
    theme = "dark" if is_dark_mode else "light"
    apply_theme(theme)

#conversion logic
#modified this so that error message resets after a success and hadles the edge cases better
def convert_currency():
    #definig them
    from_curr = entry_from.get().upper().strip()
    to_curr = entry_to.get().upper().strip()
    amount = entry_amount.get().strip()

    if from_curr in ["", PLACEHOLDER_FROM] or to_curr in ["", PLACEHOLDER_TO] or amount in ["", PLACEHOLDER_AMOUNT]:
        error_label.config(text="Please fill all fields with valid values.")
        return
    
    try:
        amount = float(amount)
    except ValueError:
        error_label.config(text="Invalid amount. Please enter a valid number.")
        return
    
    #api url
    url = f"https://api.currencyfreaks.com/latest?apikey={API_KEY}&symbols={from_curr},{to_curr}"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        base_currency = data.get("base", "")
        rates = data.get("rates",{})

        #taking care of base case and falling back to usd incase either from or t_curr is the base currrency.
        rate_from = float(rates.get(from_curr)) if from_curr !=base_currency else 1.0
        rate_to = float(rates.get(to_curr)) if to_curr != base_currency else 1.0

        if not rate_from or not rate_to:
            error_label.config(text="Invalid currency codes or Unvailable rate. \n Please try again.")
            return
        
        #since it used usd as base we convert using usd as base
        usd_amount = amount/rate_from #converted to usd
        converted = round(usd_amount * rate_to, 2) #converted to to_curr

        result_label.config(text=f"{amount} {from_curr} = {converted} {to_curr}")
        error_label.config(text="")#error cleared after a success

    except requests.RequestException:
        error_label.config(text="Network error. Check your internet or API key.")
    except ValueError:
        error_label.config(text="Error processing the exchange rate.")
        #here since the earlier one (frankfurt) changed their documents (inr isn't included anymore) i had to ended switching over currencyfreaks. 
        #however to access the converstion i would have to buy the starter plan so to avoid that i removed the base case. But since it's possible 
        #that some currencies might be the base so to counter that i used the above logic that runs independent of what the base currency is and will work 
        #even in the case of the from_curr and to_curr beign one of the base currencies.


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
#modified for placeholder
entry_from = labeled_entry("From Currency Code (3 letters):", PLACEHOLDER_FROM)
entry_to = labeled_entry("To Currency Code (3 letters):", PLACEHOLDER_TO)
entry_amount = labeled_entry("Amount:", PLACEHOLDER_AMOUNT)

#convertion button
convert_button = tk.Button(root, text="Convert", command=convert_currency, font=FONT_LABEL, width=15)
convert_button.pack(pady=15)

#result label:
result_label = tk.Label(root, text="", font=FONT_RESULT)
result_label.pack(pady=15)

#error label:
error_label = tk.Label(root, text="", fg="red", font=("Helvetica", 10, "italic"))
error_label.pack(pady=(5, 0))

#toggling button
toggle_button = tk.Button(root, text="Toggle Theme", command=toggle_theme, font=("Helvetica", 9, "bold"))
toggle_button.pack(pady=5)

apply_theme("light")

#to start the conversion (instead onf clicking on convert just simply pressing enter):
root.bind("<Return>", lambda event:convert_currency())

root.mainloop()
