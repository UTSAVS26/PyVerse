import requests
import tkinter as tk
from tkinter import messagebox
import webbrowser

def find_recipes(ingredients):
    SPOONACULAR_API_KEY = 'feb01116afe3415cb6cca75636a799d8'
    url = f'https://api.spoonacular.com/recipes/findByIngredients?ingredients={ingredients}&apiKey={SPOONACULAR_API_KEY}'
    
    response = requests.get(url)

    if response.status_code == 200:
        return response.json()
    else:
        messagebox.showerror("Error", "Failed to retrieve recipes.")
        return []

def open_recipe(url):
    webbrowser.open(url)

def display_recipes(recipes):
    # Create a popup window with a specified size
    recipe_window = tk.Toplevel()
    recipe_window.title("Recipe Results")
    recipe_window.geometry("500x400")  # Set the size of the popup window (width x height)
    
    for recipe in recipes:
        recipe_name = recipe['title']
        recipe_url = f"https://spoonacular.com/recipes/{recipe['id']}-{recipe_name.replace(' ', '-')}"
        
        recipe_link = tk.Label(recipe_window, text=recipe_name, fg="blue", cursor="hand2")
        recipe_link.pack(pady=10)
        
        # Bind click event to open the recipe URL
        recipe_link.bind("<Button-1>", lambda e, url=recipe_url: open_recipe(url))

def search_recipes():
    ingredients = ingredients_entry.get()  # Get ingredients from the entry field
    if ingredients:
        recipes = find_recipes(ingredients)
        display_recipes(recipes)

# Create the main window
root = tk.Tk()
root.title("Recipe Finder")
root.geometry("500x300")  # Increase the size of the main window

# Set a colorful background
root.configure(bg="#6A5ACD")  

# Ingredients input
tk.Label(root, text="Enter ingredients (comma-separated):", bg="#ffffff", font=("Helvetica", 14)).pack(pady=10)
ingredients_entry = tk.Entry(root, width=40, font=("Helvetica", 14), bg="#FFE4B5")  # Light lemon chiffon color
ingredients_entry.pack(pady=50)

# Search button
search_button = tk.Button(root, text="Find Recipes", command=search_recipes, bg="#90EE90", font=("Helvetica", 14))  # Light green color
search_button.pack(pady=20)

root.mainloop()
