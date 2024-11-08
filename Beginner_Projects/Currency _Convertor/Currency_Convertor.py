import tkinter as tk
from tkinter import messagebox
import requests

# GUI
root = tk.Tk()
root.title("Currency Converter")
root.configure(bg='#f2f2f2')  # Set a light background color

# Function to fetch exchange rates
def fetch_exchange_rate(from_currency, to_currency):
    api_key = '0a026bcfdce7292f2d1e7ca8'  # Your API key
    url = f'https://v6.exchangerate-api.com/v6/{api_key}/latest/{from_currency}'
    
    try:
        response = requests.get(url)
        data = response.json()

        if 'conversion_rates' in data:
            return data['conversion_rates'].get(to_currency)
        else:
            return None
    except requests.exceptions.RequestException as e:
        messagebox.showerror("Error", f"An error occurred: {e}")
        return None

# Function to perform currency conversion
def RealTimeCurrencyConversion():
    from_currency = variable1.get()
    to_currency = variable2.get()

    if not Amount1_field.get():
        messagebox.showinfo("Error !!", "Amount Not Entered.\n Please enter a valid amount.")
        return

    if from_currency == "currency" or to_currency == "currency":
        messagebox.showinfo("Error !!", "Currency Not Selected.\n Please select FROM and TO Currency from menu.")
        return

    rate = fetch_exchange_rate(from_currency, to_currency)
    if rate is None:
        messagebox.showinfo("Error !!", "Could not fetch conversion rate.")
        return

    new_amt = rate * float(Amount1_field.get())
    Amount2_field.delete(0, tk.END)
    Amount2_field.insert(0, str(round(new_amt, 4)))

# Function to clear all input fields
def clear_all():
    Amount1_field.delete(0, tk.END)
    Amount2_field.delete(0, tk.END)

# Currency list with more currencies
CurrenyCode_list = ["INR", "USD", "CAD", "CNY", "DKK", "EUR", "GBP", "AUD", "JPY", "CHF", "NZD", "SGD", "ZAR", "HKD", "NOK", "SEK"]

root.geometry("700x400")
variable1 = tk.StringVar(root)
variable2 = tk.StringVar(root)

variable1.set("currency")
variable2.set("currency")

# UI Elements
label1 = tk.Label(root, text="Amount:", bg='#f2f2f2', font=('Arial', 14))
label1.grid(row=0, column=0, sticky=tk.W)

Amount1_field = tk.Entry(root, font=('Arial', 14), width=15)
Amount1_field.grid(row=0, column=1)

FromCurrency_option = tk.OptionMenu(root, variable1, *CurrenyCode_list)
FromCurrency_option.config(bg='#4CAF50', fg='white')  # Green background
FromCurrency_option.grid(row=1, column=1, sticky=tk.W)

label2 = tk.Label(root, text="From Currency:", bg='#f2f2f2', font=('Arial', 14))
label2.grid(row=1, column=0, sticky=tk.W)

ToCurrency_option = tk.OptionMenu(root, variable2, *CurrenyCode_list)
ToCurrency_option.config(bg='#4CAF50', fg='white')  # Green background
ToCurrency_option.grid(row=2, column=1, sticky=tk.W)

label3 = tk.Label(root, text="To Currency:", bg='#f2f2f2', font=('Arial', 14))
label3.grid(row=2, column=0, sticky=tk.W)

Amount2_field = tk.Entry(root, font=('Arial', 14), width=15)
Amount2_field.grid(row=3, column=1)

label4 = tk.Label(root, text="Converted Amount:", bg='#f2f2f2', font=('Arial', 14))
label4.grid(row=3, column=0, sticky=tk.W)

convert_button = tk.Button(root, text="Convert", command=RealTimeCurrencyConversion, bg='#2196F3', fg='white', font=('Arial', 14))
convert_button.grid(row=4, column=0)

clear_button = tk.Button(root, text="Clear All", command=clear_all, bg='#F44336', fg='white', font=('Arial', 14))
clear_button.grid(row=4, column=1)

# Add some padding and spacing
for widget in root.winfo_children():
    widget.grid_configure(padx=10, pady=10)

root.mainloop()
