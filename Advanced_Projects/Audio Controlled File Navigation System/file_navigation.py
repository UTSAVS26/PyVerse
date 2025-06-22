from pathlib import Path
import os
import json
import pyautogui
import sys
import subprocess
import shutil

class FileNavigator:
    def __init__(self):
        self.current_dir = Path.home()
        self.commands = ["open", "show", "go to", "navigate to", 
                         "list", "up", "down", "rename", 
                         "delete", "create", 
                         "close"]
        
        self.system_folders = {
            "downloads": str(Path.home() / "Downloads"),
            "documents": str(Path.home() / "Documents"),
            "desktop": str(Path.home() / "Desktop"),
            "pictures": str(Path.home() / "Pictures"),
            "music": str(Path.home() / "Music"),
            "videos": str(Path.home() / "Videos"),
        }

    def _open_cross_platform(self, path):
        if sys.platform == "win32":
            os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.run(["open", str(path)])
        else:
            subprocess.run(["xdg-open", str(path)])

    def list_dir(self, path=None):
        path = Path(path) if path else self.current_dir
        if path.exists() and path.is_dir():
            print(f"Contents of {path}:")
            for p in path.iterdir():
                print(p.name)
        else:
            print(f"Directory not found: {path}")

    def open_path(self, path):
        path = Path(path)
        if path.exists():
            self._open_cross_platform(path)
            print(f"Opened: {path}")
            self.current_dir=path
        else:
            print(f"Path not found: {path}")

    def navigate_up(self):
        self.current_dir = self.current_dir.parent
        self._open_cross_platform(self.current_dir)

    def navigate_down(self, folder):
        new_dir = self.current_dir / folder
        if new_dir.exists() and new_dir.is_dir():
            self.current_dir = new_dir
            print(f"Moved down to: {self.current_dir}")
            self._open_cross_platform(self.current_dir)
        else:
            print(f"Folder not found: {folder}")

    def rename(self, text):
        parts = text.split()
        if "rename" in parts and len(parts) >= 3:
            idx = parts.index("rename")
            if idx + 2 < len(parts):
                old_name = parts[idx+1]
                new_name = parts[idx+2]
                old_path = self.current_dir / old_name
                new_path = self.current_dir / new_name
                if old_path.exists():
                    old_path.rename(new_path)
                    print(f"Renamed {old_name} to {new_name}")
                else:
                    print(f"File/Folder not found: {old_name}")
                return
        print("Invalid rename command format. Use: rename <old> <new>")

    def delete(self, text):
        parts = text.split()
        if "delete" in parts and len(parts) > 1:
            idx = parts.index("delete")

            if idx + 1 < len(parts):
                name = parts[idx+1]
                path = self.current_dir / name

                if path.exists():

                    if path.is_dir():
                        confirm = input(f"Delete directory '{name}' and all its contents? [y/N]: ")

                        if confirm.lower() == 'y':
                            shutil.rmtree(path)
                            print(f"Deleted: {name}")

                        else:
                            print("Deletion cancelled.")
                    else:
                        path.unlink()
                        print(f"Deleted: {name}")

                else:
                    print(f"File/Folder not found: {name}")

                return
        print("No file or folder specified to delete.")

    def create(self, text):
        parts = text.split()
        if "create" in parts and len(parts) > 1:
            idx = parts.index("create")

            if idx + 1 < len(parts):
                name = parts[idx+1]
                is_dir = "folder" in text or "directory" in text
                path = self.current_dir / name

                if not path.exists():
                    if is_dir:
                        path.mkdir()
                        print(f"Directory created: {name}")
                    else:
                        path.touch()
                        print(f"File created: {name}")

                else:
                    print(f"File/Folder already exists: {name}")
                    
                return
        print("No file or folder specified to create.")

    def _matches_command(self, text, keywords):
        return any(keyword in text for keyword in keywords)

    def parse_command(self, text):
        try:
            text_dict = json.loads(text)
            recognized_text = text_dict['text'].lower().strip()
            print(f"Recognized: {recognized_text}")

            if self._matches_command(recognized_text, ['list']):
                self.list_dir()

            elif self._matches_command(recognized_text, ['up']):
                self.navigate_up()

            elif self._matches_command(recognized_text, ['down']):
                self.navigate_down(recognized_text)

            elif self._matches_command(recognized_text, ["open", "show", "go to", "navigate to"]):
                self.open_path(recognized_text)

            elif self._matches_command(recognized_text, ["rename"]):
                self.rename(recognized_text)

            elif self._matches_command(recognized_text, ["delete"]):
                self.delete(recognized_text)

            elif self._matches_command(recognized_text, ["create"]):
                self.create(recognized_text)
                
            elif self._matches_command(recognized_text, ["close"]):
                pyautogui.hotkey('alt', 'f4')
                print("Closed current window.")
                self.current_dir = Path.home()
            else:
                print("Unrecognized command.")
        except json.JSONDecodeError:
            print("Error: Invalid JSON format in recognized text")
        except Exception as e:
            print(f"Error: {e}") 
