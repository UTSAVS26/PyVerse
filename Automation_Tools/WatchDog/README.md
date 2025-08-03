# WatchDog - Watch Library

## Introduction

WatchDog is a simple file monitoring utility built using Python and the `watchdog` library. It watches a specified folder (configurable, defaults to Downloads) for new or modified files whose names end with a configurable pattern (defaults to `-watchdog` before the file extension). When such files are detected, they are automatically moved to a target folder (configurable, defaults to `Documents/watchdog_files`).

## Features

- Monitors a folder for file creation, modification, or movement.
- Automatically moves files ending with a configurable pattern to a specified directory.
- Uses Python's `shutil` library to move files efficiently and safely.
- Easy to configure source and target folders.
- Cross-platform path handling using `os.path.expanduser`.
- Handles file conflicts by automatically renaming duplicates.
- Comprehensive error handling for permissions and OS errors.

## Requirements

- Python 3.x
- `watchdog` library

Install dependencies with:

```bash
pip install watchdog
```

## Usage

1. Edit `main.py` to set your desired `FILE_PATTERN`, `watch_folder` and `target_folder` paths if needed.
1. Run the script:

```bash
python main.py
```

1. The script will keep running, watching for files. Press `Ctrl+C` to stop.

## Customization

- Change the `FILE_PATTERN` variable to match your desired filename suffix (without the hyphen).
- Change the `watch_folder` and `target_folder` variables in `main.py` to suit your needs.
- The script only moves files whose names end with the configured pattern (defaults to `-watchdog` before the extension).

## How it works

- The script uses the `watchdog` library to monitor file system events in the specified folder.
- When a file event is detected, it checks if the filename ends with the configured pattern.
- If it matches, the script uses Python's `shutil.move()` function to move the file to the target directory.
- If a file with the same name already exists, it automatically appends a counter to avoid conflicts.

## Configuration Examples

- To watch for files ending with `-myfiles`: Set `FILE_PATTERN = "myfiles"`
- To use a different watch folder: Set `watch_folder = os.path.expanduser("~/Desktop")`
- To use a different target folder: Set `target_folder = os.path.expanduser("~/Documents/my_sorted_files")`

## License

This project is for personal use and learning purposes.
