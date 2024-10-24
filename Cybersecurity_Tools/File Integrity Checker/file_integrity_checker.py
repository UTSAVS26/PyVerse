import hashlib
import os
import time
import logging
import threading
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
from colorama import Fore, Style

# Set up logging
logging.basicConfig(filename='file_integrity.log', level=logging.INFO)

# Global variable to control monitoring state
monitoring = False

def calculate_hash(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def log_change(message, change_type, file_size=None):
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    if file_size:
        message += f" | Size: {file_size} bytes"
    log_message = f"{timestamp} | {message} | Change type: {change_type}"
    
    if change_type == "new":
        print(Fore.GREEN + log_message + Style.RESET_ALL)
    elif change_type == "changed":
        print(Fore.YELLOW + log_message + Style.RESET_ALL)
    elif change_type == "deleted":
        print(Fore.RED + log_message + Style.RESET_ALL)
    
    logging.info(log_message)

def monitor_directory(directory, scan_interval, gui_msg):
    global monitoring
    file_hashes = {}

    while monitoring:
        current_files = {}
        new_files = 0
        changed_files = 0
        deleted_files = 0

        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                file_hash = calculate_hash(file_path)
                file_size = os.path.getsize(file_path)
                current_files[file_path] = file_hash

                # Compare current files with previous hashes
                if file_path not in file_hashes:
                    log_change(f"New file detected: {file_path}", "new", file_size)
                    new_files += 1
                elif file_hashes[file_path] != file_hash:
                    log_change(f"File changed: {file_path}", "changed", file_size)
                    changed_files += 1

        # Removed files
        removed_files = set(file_hashes.keys()) - set(current_files.keys())
        for file_path in removed_files:
            log_change(f"File deleted: {file_path}", "deleted")
            deleted_files += 1

        file_hashes = current_files
        gui_msg.set(f"New files: {new_files}, Changed files: {changed_files}, Deleted files: {deleted_files}")
        time.sleep(scan_interval)

def start_monitoring(directory, scan_interval, gui_msg):
    global monitoring
    monitoring = True
    gui_msg.set("Monitoring started...")
    monitor_thread = threading.Thread(target=monitor_directory, args=(directory, scan_interval, gui_msg))
    monitor_thread.start()

def stop_monitoring(gui_msg):
    global monitoring
    monitoring = False
    gui_msg.set("Monitoring stopped...")

# Tkinter GUI
def select_directory():
    directory = filedialog.askdirectory()
    directory_entry.delete(0, END)
    directory_entry.insert(0, directory)

def start_monitoring_gui():
    directory = directory_entry.get()
    if not directory:
        messagebox.showwarning("Input Error", "Please select a directory to monitor.")
        return
    
    try:
        scan_interval = int(scan_interval_entry.get())
    except ValueError:
        messagebox.showwarning("Input Error", "Please enter a valid scan interval (in seconds).")
        return
    
    start_monitoring(directory, scan_interval, gui_msg)

def stop_monitoring_gui():
    stop_monitoring(gui_msg)

# Create the main window
root = Tk()
root.title("File Integrity Checker")
root.geometry("400x250")

# Directory selection
directory_label = Label(root, text="Directory to Monitor:")
directory_label.pack(pady=5)
directory_entry = Entry(root, width=50)
directory_entry.pack(pady=5)
directory_button = Button(root, text="Select Directory", command=select_directory)
directory_button.pack(pady=5)

# Scan interval input
scan_interval_label = Label(root, text="Scan Interval (in seconds):")
scan_interval_label.pack(pady=5)
scan_interval_entry = Entry(root, width=10)
scan_interval_entry.pack(pady=5)

# Start and Stop monitoring buttons
start_button = Button(root, text="Start Monitoring", command=start_monitoring_gui)
start_button.pack(pady=5)
stop_button = Button(root, text="Stop Monitoring", command=stop_monitoring_gui)
stop_button.pack(pady=5)

# Message area
gui_msg = StringVar()
message_label = Label(root, textvariable=gui_msg)
message_label.pack(pady=10)

# Run the Tkinter event loop
root.mainloop()
