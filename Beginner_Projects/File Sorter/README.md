# File Organizer

File Organizer is a simple Python script and GUI application for organizing files in a directory based on their file extensions. It allows you to specify a directory, file extensions, and an optional log file for tracking file movements.

----

![image](https://github.com/SpreadSheets600/File-Sorter/assets/115402296/5907a1d1-548f-48da-81d7-1ec352a9c57b)

----

## Features

- Organize files in a directory based on their file extensions.
- Log file movements with timestamps.
- Undo file organization.
- Choose between light and dark themes.

## Getting Started

These instructions will help you get a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

- Python 3.x
- Tkinter (usually included with Python, no separate installation required)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/file-organizer.git
```

2. Change into the project directory:

```bash
cd file-organizer
```

3. Ensure the Azure-ttk-theme folder is in the correct location, as it is required for the application's theme.
   Without the theme, the application will not work. (Make changes to line number 117 And 120)

```python
window.iconbitmap(r"")
window.resizable(width=True, height=True) # App Icon Location

window.tk.call("source", r"\Theme\azure.tcl")
window.tk.call("set_theme", "light") # `azure.tcl` Location From Theme Folder
```

4. Run the application:

```bash
python App.py
```

## Usage

1. Launch the application by running 'App.py`.

2. Provide the following information in the GUI:
   - Directory Path: The directory you want to organize.
   - File Extensions: Comma-separated file extensions (e.g., `pdf, txt, docx`).
   - Log File (Optional): Path to a log file for tracking file movements.

3. Click the "Organize Files" button to start the organization process.

4. You can undo the file organization by clicking the "Undo" button.

5. Change the theme between light and dark with the "Theme" button.


## Contributing

If you would like to contribute to this project, you can follow these steps:

1. Fork the repository.
2. Create a new branch for your feature: `git checkout -b feature-name`
3. Make your changes and commit them: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature-name`
5. Create a pull request.


## Acknowledgments

- Thanks to the Python community for creating powerful libraries and tools.
- Special thanks to contributors.
