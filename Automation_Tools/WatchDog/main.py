import os
import shutil
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Folder to watch
watch_folder = r"C:\Users\Shravan\Downloads"

# Folder to move files to
target_folder = r"C:\Users\Shravan\Documents\shravanfiles"

# Create the target folder if it doesn't exist
os.makedirs(target_folder, exist_ok=True)

# File watcher handler
class MoveShravanFilesHandler(FileSystemEventHandler):
    def process(self, event):
        if event.is_directory:
            return

        file_path = event.src_path
        file_name = os.path.basename(file_path)
        file_base, file_ext = os.path.splitext(file_name)

        # Check if filename ends with -shravan (before extension)
        if file_base.endswith("-shravan"):
            dest_path = os.path.join(target_folder, file_name)
            try:
                shutil.move(file_path, dest_path)
                print(f"Moved: {file_name}")
            except Exception as e:
                print(f"Failed to move {file_name}: {e}")

    def on_created(self, event):
        self.process(event)

    def on_modified(self, event):
        self.process(event)

    def on_moved(self, event):
        self.process(event)

# Main setup
if __name__ == "__main__":
    observer = Observer()
    handler = MoveShravanFilesHandler()
    observer.schedule(handler, watch_folder, recursive=False)
    observer.start()

    print("Watching for *-shravan.* files in Downloads folder... (Ctrl+C or close VS Code to stop)")

    try:
        while True:
            pass  # Keeps the watcher running
    except KeyboardInterrupt:
        print("\nStopping watcher...")
        observer.stop()
    observer.join()
