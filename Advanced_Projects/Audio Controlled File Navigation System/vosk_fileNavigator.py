from pathlib import Path
import queue
import sounddevice as sd
from vosk import Model, KaldiRecognizer
import pyaudio
import os
import json
import pyautogui

# Load model using pathlib
model_path = Path.cwd() / "vosk-model-en-in-0.5"
assert model_path.exists(), f"Model not found at {model_path}"

model = Model(str(model_path))
recognizer = KaldiRecognizer(model, 16000)
audio_queue = queue.Queue()

# Callback to collect audio chunks
def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(bytes(indata))

class FileNavigator:
    def __init__(self):
        self.current_dir = Path.home()
        self.commands = ["open", "show", "go to", "navigate to", 
                         "list", "up", "down", "rename", 
                         "delete", "create", 
                         "close"]
        #add more commands and make their functions to increase and add functionality
        
        self.system_folders = {
            "downloads": str(Path.home() / "Downloads"),
            "documents": str(Path.home() / "Documents"),
            "desktop": str(Path.home() / "Desktop"),
            "pictures": str(Path.home() / "Pictures"),
            "music": str(Path.home() / "Music"),
            "videos": str(Path.home() / "Videos"),
            #add more appropriate folder paths that you want to use
        }


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
            os.startfile(path)
            print(f"Opened: {path}")
            self.current_dir=path;
        else:
            print(f"Path not found: {path}")




    def navigate_up(self):
        self.current_dir = self.current_dir.parent
        os.startfile(self.current_dir)
        




    def navigate_down(self, folder):
        new_dir = self.current_dir / folder
        if new_dir.exists() and new_dir.is_dir():
            self.current_dir = new_dir
            print(f"Moved down to: {self.current_dir}")
            os.startfile(self.current_dir)
        else:
            print(f"Folder not found: {folder}")




    def rename(self, old, new):
        old_path = self.current_dir / old
        new_path = self.current_dir / new
        if old_path.exists():
            old_path.rename(new_path)
            print(f"Renamed {old} to {new}")
        else:
            print(f"File/Folder not found: {old}")




    def delete(self, name):
        path = self.current_dir / name
        if path.exists():
            if path.is_dir():
                for sub in path.iterdir():
                    if sub.is_file():
                        sub.unlink()
                    else:
                        print(f"Skipping subfolder: {sub}")
                path.rmdir()
            else:
                path.unlink()
            print(f"Deleted: {name}")
        else:
            print(f"File/Folder not found: {name}")




    def create(self, name, is_dir=False):
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





    def parse_command(self, text):
        try:
            text_dict = json.loads(text)
            recognized_text = text_dict['text'].lower().strip()
            print(f"Recognized: {recognized_text}")

            if 'list' in recognized_text:
                self.list_dir()
                
            elif 'up' in recognized_text:
                self.navigate_up()

            elif "down" in recognized_text:
                folder = recognized_text.split()[-1];
                if folder != 'down':
                    self.navigate_down(folder)

            elif any(word in recognized_text for word in ["open", "show", "go to", "navigate to"]):
                for folder, path in self.system_folders.items():
                    if folder in recognized_text:
                        self.open_path(path)
                        return
                    
                # Try to open file/folder in current directory
                parts = recognized_text.split()
                for i, word in enumerate(parts):
                    if word in ["open", "show", "go", "navigate"] and i+1 < len(parts):
                        self.open_path(self.current_dir / parts[i+1])
                        return
                    
            elif "rename" in recognized_text:
                newName = recognized_text.split()[-1]
                oldName = recognized_text.split()[-2]
                if(oldName!='rename'):
                    self.rename(oldName, newName)


            elif "delete" in recognized_text:
                name = recognized_text.split()[-1]
                if name != 'delete':
                    self.delete(name)
                else:
                    print(f"\nContent to be deleted not Found")


            elif "create" in recognized_text:
                name = recognized_text.split()[-1];
                if name != 'create':
                    is_dir = "folder" in recognized_text or "directory" in recognized_text
                    self.create(name, is_dir)


            elif "close" in recognized_text:
                pyautogui.hotkey('alt', 'f4')
                print("Closed current window.")
                self.current_dir = Path.home()

        except Exception as e:
            print(f"Error: {e}")



navigator = FileNavigator()

# Modify the main processing loop to use the command parser
command_parser = FileNavigator()

# Open microphone stream
with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                       channels=1, callback=audio_callback):
    print("🎤 Speak into the mic (Ctrl+C to stop):")
    try:
        while True:
            data = audio_queue.get()
            if recognizer.AcceptWaveform(data):
                result = recognizer.Result()
                # Parse the recognized text
                print(result)
                command_parser.parse_command(result)
                
    except KeyboardInterrupt:
        print("\n🛑 Stopped.")

p = pyaudio.PyAudio()
