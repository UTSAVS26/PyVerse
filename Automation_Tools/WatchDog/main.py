import os
import shutil
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Configurable pattern to match (without the hyphen prefix)
FILE_PATTERN = "watchdog"  # Files ending with "-watchdog" will be moved

# Folder to watch (configurable)
watch_folder = os.path.expanduser("~/Downloads")

# Folder to move files to (configurable)
target_folder = os.path.expanduser("~/Documents/watchdog_files")

# Validate and create folders
def setup_folders():
    if not os.path.exists(watch_folder):
        raise FileNotFoundError(f"Source folder does not exist: {watch_folder}")
    if not os.access(watch_folder, os.R_OK):
        raise PermissionError(f"No read permission for source folder: {watch_folder}")
    
    os.makedirs(target_folder, exist_ok=True)
    if not os.access(target_folder, os.W_OK):
        raise PermissionError(f"No write permission for target folder: {target_folder}")

setup_folders()

# File watcher handler
class FileWatcherHandler(FileSystemEventHandler):
    def process(self, event):
        if event.is_directory:
            return

        file_path = event.src_path
        file_name = os.path.basename(file_path)
        file_base, file_ext = os.path.splitext(file_name)

        # Check if filename ends with the configured pattern (before extension)
        if file_base.endswith(f"-{FILE_PATTERN}"):
            dest_path = os.path.join(target_folder, file_name)
            try:
                # Check if destination file already exists
                if os.path.exists(dest_path):
                    base, ext = os.path.splitext(dest_path)
                    counter = 1
                    while os.path.exists(f"{base}_{counter}{ext}"):
                        counter += 1
                    dest_path = f"{base}_{counter}{ext}"
                
                shutil.move(file_path, dest_path)
                print(f"Moved: {file_name}")
            except PermissionError as e:
                print(f"Permission denied moving {file_name}: {e}")
            except OSError as e:
                print(f"OS error moving {file_name}: {e}")
            except Exception as e:
                print(f"Unexpected error moving {file_name}: {e}")

    def on_created(self, event):
        self.process(event)

    def on_modified(self, event):
        self.process(event)

    def on_moved(self, event):
        self.process(event)

# Main setup
if __name__ == "__main__":
    observer = Observer()
    handler = FileWatcherHandler()
    observer.schedule(handler, watch_folder, recursive=False)
    observer.start()

    print(f"Watching for *-{FILE_PATTERN}.* files in {watch_folder}... (Ctrl+C or close VS Code to stop)")

    try:
        while True:
            pass  # Keeps the watcher running
    except KeyboardInterrupt:
        print("\nStopping watcher...")
        observer.stop()
    observer.join()
