# Recipe Finder Using Tkinter

## Overview

This project is a simple **Recipe Finder** built using **Tkinter**, a GUI library in Python. The application allows users to input ingredients they have at home and find suitable recipes. It uses the Spoonacular API to fetch and display recipes based on the provided ingredients.

## Features

- **Graphical User Interface (GUI):** The interface allows users to easily input ingredients and view corresponding recipe suggestions.
- **Recipe Search:** Users can enter a list of ingredients, and the application retrieves recipes that can be made with those ingredients.
- **Clickable Recipe Links:** The application provides links to actual recipes on the Spoonacular website, allowing users to view detailed instructions and ingredients.

## How It Works

1. **Spoonacular API:**
    - The application utilizes the Spoonacular API to access a large database of recipes. The API requires an API key for access, which must be included in the code.

2. **Tkinter GUI:**
    - The GUI is built using `Tkinter`, featuring an input field for entering ingredients and a button to initiate the recipe search.
    - Upon clicking the "Find Recipes" button, the application sends a request to the Spoonacular API with the specified ingredients.

3. **Displaying Recipes:**
    - The results are displayed in a new window, showing the titles of the recipes. Each recipe title is a clickable link that opens the corresponding recipe in a web browser.

## How to Run the Project

1. Clone the repository or download the project files.
2. Ensure you have Python 3.x installed on your machine.
3. Install the required libraries using pip:
    ```bash
    pip install requests
    ```
4. Replace the placeholder API key in the script with your Spoonacular API key.
5. Run the `recipe_finder.py` script using Python:
    ```bash
    python recipe_finder.py
    ```
6. Enter ingredients separated by commas in the input box and click "Find Recipes." The application will display recipe suggestions.

## Requirements

- Python 3.x
- Tkinter library (included with Python)
- `requests` library (install via pip)

## Future Enhancements

- **Save Favorite Recipes:** Allow users to save their favorite recipes for quick access.
- **Ingredient Suggestions:** Provide suggestions for recipes based on partial ingredients or commonly available items.
- **User Accounts:** Implement user accounts to save preferences and custom ingredient lists.
