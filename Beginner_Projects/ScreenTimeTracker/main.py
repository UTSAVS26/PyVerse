import threading
from tracker import log_screen_time
from ui import launch_ui

if __name__ == "__main__":
    t = threading.Thread(target=log_screen_time, daemon=True)
    t.start()
    launch_ui()