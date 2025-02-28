# Spell Checker Using Tkinter

## Overview

This project is a simple **Spell Checker** built using **Tkinter**, a GUI library in Python. The spell checker compares user-inputted words with a list of pre-defined words stored in a file named `words.txt`. If a word entered by the user doesn't match any word in the list, it is flagged as a misspelled word.

## Features

- **Graphical User Interface (GUI):** Built using Tkinter, the interface allows users to input text and check for spelling mistakes.
- **Text Comparison:** Words inputted by the user are compared against a predefined list of 300 words stored in `words.txt`.
- **Misspelled Word Highlighting:** If the user inputs a misspelled word, the program informs the user that the word is not found in the word list.

## How It Works

1. **`words.txt` File:** 
    - A text file (`words.txt`) contains a list of 300 common English words. This file serves as the dictionary that the program uses to verify the spelling of words.
    
    Example of a few words from `words.txt`:
    ```
    apple
    banana
    computer
    python
    technology
    ...
    ```
    The complete file includes a wide range of words covering categories such as animals, objects, professions, and more.

2. **Tkinter GUI:**
    - A GUI is designed with `Tkinter`, allowing the user to input a sentence. The program then checks each word against the list in `words.txt`.
    - The result will display whether each word is correct or if it is misspelled.

3. **Spell Checker Logic:**
    - The program reads the `words.txt` file and stores the words in a list.
    - The input provided by the user is split into individual words.
    - Each word is compared with the list of words from `words.txt`. If the word is not found, the program flags it as a misspelled word.


## How to Run the Project

1. Clone the repository or download the project files.
2. Make sure the `words.txt` file is in the same directory as the Python script.
3. Run the `spell_checker.py` script using Python:
    ```
    python spell_checker.py
    ```
4. Enter a sentence or a group of words in the input field of the GUI and click "Check Spelling."
5. The program will highlight any misspelled words.

## Requirements

- Python 3.x
- Tkinter library (included with Python)

## Future Enhancements

- **Dynamic Word Suggestions:** Add functionality to suggest correct words for misspelled words.
- **Custom Dictionary:** Allow users to add words to their custom dictionary.
- **Advanced Spell Check:** Include handling of punctuation and more advanced grammar rules.




