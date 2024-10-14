
# File Integrity Checker

This Python tool monitors file changes (new, modified, deleted) in a specified directory. It calculates the SHA-256 hash of each file to detect changes and logs the details of any changes. It also features a simple GUI built using Tkinter.

## Features

- **Monitor directory:** Continuously monitors a chosen directory for new, changed, or deleted files.
- **SHA-256 checksum:** Uses a cryptographic hash function to detect changes.
- **Logging:** Logs the detected file changes in a file named `file_integrity.log`.
- **Colored console output:** Uses the `colorama` library for colored outputs (green for new, yellow for changed, red for deleted).
- **GUI:** A user-friendly interface for directory selection and control using Tkinter.

## Requirements

- Python 3.x
- Libraries: Install the necessary dependencies using pip:

    ```bash
    pip install colorama
    ```

    Tkinter is usually included with Python, but if needed, it can be installed separately based on your operating system.

## How to Use

1. **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2. **Run the script:**

    ```bash
    python file_integrity_checker.py
    ```

3. **Select a directory to monitor:**
    - Click the "Select Directory" button in the GUI to choose the directory you want to monitor.
  
4. **Set scan interval:**
    - Enter the scan interval (in seconds) for how frequently the directory should be checked.

5. **Start monitoring:**
    - Click the "Start Monitoring" button to begin checking for changes in the directory.
  
6. **Stop monitoring:**
    - Click the "Stop Monitoring" button to stop the monitoring process.

## Integration in Other Projects

To integrate this tool into other projects:
1. Import necessary functions like `start_monitoring()` and `stop_monitoring()` from this script.
2. Set up the monitoring in your project's logic by calling these functions as needed.


## Logging

- All changes (new, modified, deleted files) will be logged in `file_integrity.log`.
  
