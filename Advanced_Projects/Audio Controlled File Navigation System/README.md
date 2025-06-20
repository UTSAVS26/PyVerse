# Audio Controlled File Navigation System

This project allows you to control file navigation and basic file operations on your computer using voice commands. It leverages both the Vosk speech recognition toolkit and OpenAI's Whisper model (via faster-whisper) to interpret spoken commands and perform actions such as opening folders, listing directory contents, navigating directories, and basic file operations (rename, delete, create).

## Features
- Open folders or files via speech commands
- List directory contents
- Navigate up/down directories
- Basic file operations: rename, delete, create
- Close the current window using voice

## Setup Instructions

### 1. Clone the Repository
```
git clone https://github.com/UjjWaL-0911/PyVerse.git
cd Audio-Controlled-File-Navigation-System
```

### 2. Install Python Dependencies
Make sure you have Python 3.7+ installed. Then run:
```
pip install -r requirements.txt
```

---

## Vosk Model Setup

### Download and Setup Vosk Model
1. Download the Vosk English model (e.g., `vosk-model-en-in-0.5`) from the [Vosk Models page](https://alphacephei.com/vosk/models).
2. Extract the downloaded model folder into your project directory so that you have:
   ```
   Audio Controlled File Navigation System/
     vosk-model-en-in-0.5/
       am/
       conf/
       ...
   ```
3. The code expects the model folder to be named exactly `vosk-model-en-in-0.5` in the project root. If you use a different model, update the path in the code accordingly.

### Run the Vosk Application
```
python vosk_fileNavigator.py
```

---

## Whisper Model Setup

### Download and Setup Whisper Model
- The `faster-whisper` library will automatically download the required Whisper model (e.g., `medium.en`) the first time you run the script. You can change the model size by editing the `model_size` variable in `whisper_fileNavigator.py`.
- For best performance, a CUDA-compatible GPU is recommended. If you do not have one, set `device="cpu"` in the script.

### Run the Whisper Application
```
python whisper_fileNavigator.py
```

---

## Testing Inference (How to Speak Commands)

### Listing Directory Contents
- **Say:** `list`
- **Effect:** Lists the contents of the current directory.

### Navigating Up a Directory
- **Say:** `up`
- **Effect:** Moves up to the parent directory and opens it.

### Navigating Down into a Subfolder
- **Say:** `down <foldername>`
  - Example: `down Documents`
- **Effect:** Moves into the specified subfolder and opens it.

### Opening a System Folder
- **Say:** `open downloads`, `open documents`, `open desktop`, etc.
- **Effect:** Opens the specified system folder.

### Opening a File or Folder in Current Directory
- **Say:** `open <name>`
  - Example: `open cat.txt`
- **Effect:** Opens the specified file or folder in the current directory.

### Renaming a File or Folder
- **Say:** `rename <oldname> <newname>`
  - Example: `rename cat.txt dog.txt`
- **Effect:** Renames `cat.txt` to `dog.txt` in the current directory.

### Deleting a File or Folder
- **Say:** `delete <name>`
  - Example: `delete cat.txt`
- **Effect:** Deletes the specified file or folder in the current directory.

### Creating a File or Folder
- **Say:** `create <name>` (for a file)
  - Example: `create notes.txt`
- **Say:** `create folder <name>` or `create directory <name>` (for a folder)
  - Example: `create folder projects`
- **Effect:** Creates a file or folder in the current directory.

### Closing the Current Window
- **Say:** `close`
- **Effect:** Closes the currently focused window (via Alt+F4).

---

## Notes on Command Format
- The code extracts command arguments by splitting the recognized text and using the last or second-to-last word for file/folder names. For best results, speak commands in the formats shown above.
- Avoid extra words or filler in your commands (e.g., say `delete cat.txt` instead of `please delete cat.txt`).
- Folder and file names are case-insensitive but must match the actual names in the directory.

---

## Additional Notes for Whisper Model
- You can change the Whisper model size by editing the `model_size` variable in `whisper_fileNavigator.py`.
- If you encounter issues with CUDA, you can set `device="cpu"` in the script.
- The script prints debug information to help diagnose issues with audio input or model inference.
