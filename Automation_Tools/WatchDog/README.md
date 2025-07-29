# WatchDog - Watch Library

## Introduction
WatchDog is a simple file monitoring utility built using Python and the `watchdog` library. It watches a specified folder (by default, your Downloads folder) for new or modified files whose names end with `-filename` (before the file extension). When such files are detected, they are automatically moved to a target folder (by default, `Documents/filename`).

## Features
- Monitors a folder for file creation, modification, or movement.
- Automatically moves files ending with `-filename` to a specified directory.
- Uses Python's `shutil` library to move files efficiently and safely.
- Easy to configure source and target folders.

## Requirements
- Python 3.x
- `watchdog` library

Install dependencies with:
```
pip install watchdog
```

## Usage
1. Edit `main.py` to set your desired `watch_folder` and `target_folder` paths if needed.
2. Run the script:
```
python main.py
```
3. The script will keep running, watching for files. Press `Ctrl+C` to stop.

## Customization
- Change the `watch_folder` and `target_folder` variables in file`main.py` to suit your needs.
- The script only moves files whose names end with `-filename` (before the extension).

## How it works
- The script uses the `watchdog` library to monitor file system events in the specified folder.
- When a file event is detected, it checks if the filename ends with `-filename`.
- If it matches, the script uses Python's `shutil.move()` function to move the file to the target directory.

## License
This project is for personal use and learning purposes.
