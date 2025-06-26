# 🗂️ File Organizer GUI

A simple Python desktop application to help you organize your files easily using a graphical user interface built with Tkinter.

## 🚀 Features

- 📁 Organize files by:
  - File Type
  - Last Modified Date
  - File Extension
- 🗃️ Create subfolders automatically
- 📋 Copy files (optional) or move them
- 🔍 Preview how files will be organized before applying
- 🧾 Logs all actions for review
- 🧵 Non-blocking GUI with multithreading

## 🛠️ Tech Stack

- Python 3.x
- Tkinter (for GUI)

## 📦 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

**Note:** Tkinter is usually bundled with Python. If not, you may need to install it manually depending on your OS.

## 🖥️ How to Run

1. Clone or download the repository.
2. Run the main Python script:

```bash
python filegui.py
```

3. Use the GUI to:
   - Select a source folder with files.
   - Select a destination folder.
   - Choose your organization method and options.
   - Preview and then organize files.

## 📂 Organization Logic

- **By File Type:** Images, Documents, Videos, etc.
- **By Date Modified:** Folders by `YYYY-MM`
- **By Extension:** `.pdf`, `.jpg`, etc.

## 🧑‍💻 Author

Adwitya Chakraborty  
[https://github.com/adwityac] 