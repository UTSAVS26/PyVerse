import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import shutil
from pathlib import Path
from datetime import datetime
import threading
import queue

class FileOrganizer:
    def __init__(self, root):
        self.root = root
        self.root.title("File Organizer")
        self.root.geometry("800x600")
        self.root.configure(bg='#f0f0f0')
        
        # Variables
        self.source_folder = tk.StringVar()
        self.dest_folder = tk.StringVar()
        self.organize_method = tk.StringVar(value="type")
        
        # Thread-safe queue for UI updates
        self._queue = queue.Queue()
        
        # File type mappings
        self.file_types = {
            'Images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.svg', '.webp'],
            'Documents': ['.pdf', '.doc', '.docx', '.txt', '.rtf', '.odt', '.xls', '.xlsx', '.ppt', '.pptx'],
            'Videos': ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v'],
            'Audio': ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.wma', '.m4a'],
            'Archives': ['.zip', '.rar', '.7z', '.tar', '.gz', '.bz2'],
            'Code': ['.py', '.js', '.html', '.css', '.cpp', '.java', '.c', '.php', '.rb'],
            'Executables': ['.exe', '.msi', '.deb', '.rpm', '.dmg', '.app']
        }
        
        self.setup_ui()
        
        # Start queue processing
        self.root.after(50, self._process_queue)
        
    def _process_queue(self):
        """Process UI update callbacks from worker thread"""
        try:
            while not self._queue.empty():
                callback = self._queue.get_nowait()
                try:
                    callback()
                except Exception as exc:
                    # Keep UI alive and surface the error
                    self.log_message(f"UI callback error: {exc}")
        finally:
            # Schedule next queue check
            self.root.after(50, self._process_queue)
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title = ttk.Label(main_frame, text="File Organizer", font=('Arial', 16, 'bold'))
        title.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Source folder selection
        ttk.Label(main_frame, text="Source Folder:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.source_folder, width=50).grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_source).grid(row=1, column=2, padx=5)
        
        # Destination folder selection
        ttk.Label(main_frame, text="Destination Folder:").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.dest_folder, width=50).grid(row=2, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_dest).grid(row=2, column=2, padx=5)
        
        # Organization method
        method_frame = ttk.LabelFrame(main_frame, text="Organization Method", padding="10")
        method_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=20)
        
        ttk.Radiobutton(method_frame, text="By File Type", variable=self.organize_method, value="type").grid(row=0, column=0, sticky=tk.W, padx=10)
        ttk.Radiobutton(method_frame, text="By Date Modified", variable=self.organize_method, value="date").grid(row=0, column=1, sticky=tk.W, padx=10)
        ttk.Radiobutton(method_frame, text="By File Extension", variable=self.organize_method, value="extension").grid(row=0, column=2, sticky=tk.W, padx=10)
        
        # Options frame
        options_frame = ttk.LabelFrame(main_frame, text="Options", padding="10")
        options_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        self.copy_files = tk.BooleanVar(value=True)
        self.create_subfolders = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(options_frame, text="Copy files (keep originals)", variable=self.copy_files).grid(row=0, column=0, sticky=tk.W, padx=10)
        ttk.Checkbutton(options_frame, text="Create subfolders", variable=self.create_subfolders).grid(row=0, column=1, sticky=tk.W, padx=10)
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        # Status label
        self.status_label = ttk.Label(main_frame, text="Ready to organize files")
        self.status_label.grid(row=6, column=0, columnspan=3, pady=5)
        
        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=7, column=0, columnspan=3, pady=20)
        
        ttk.Button(button_frame, text="Preview", command=self.preview_organization).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Organize Files", command=self.organize_files).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear Log", command=self.clear_log).pack(side=tk.LEFT, padx=5)
        
        # Log area
        log_frame = ttk.LabelFrame(main_frame, text="Log", padding="5")
        log_frame.grid(row=8, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        
        self.log_text = tk.Text(log_frame, height=15, width=80)
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(8, weight=1)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
    
    def browse_source(self):
        folder = filedialog.askdirectory(title="Select Source Folder")
        if folder:
            self.source_folder.set(folder)
    
    def browse_dest(self):
        folder = filedialog.askdirectory(title="Select Destination Folder")
        if folder:
            self.dest_folder.set(folder)
    
    def log_message(self, message):
        self.log_text.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')} - {message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
    
    def clear_log(self):
        self.log_text.delete(1.0, tk.END)
    
    def get_file_category(self, file_path):
        ext = Path(file_path).suffix.lower()
        for category, extensions in self.file_types.items():
            if ext in extensions:
                return category
        return 'Others'
    
    def get_date_folder(self, file_path):
        timestamp = os.path.getmtime(file_path)
        date = datetime.fromtimestamp(timestamp)
        return f"{date.year}-{date.month:02d}"
    
    def get_extension_folder(self, file_path):
        ext = Path(file_path).suffix.lower()
        return ext[1:] if ext else 'no_extension'
    
    def get_files_to_organize(self):
        source = self.source_folder.get()
        if not source or not os.path.exists(source):
            return []
        
        files = []
        for item in os.listdir(source):
            item_path = os.path.join(source, item)
            if os.path.isfile(item_path):
                files.append(item_path)
        return files
    
    def get_destination_path(self, file_path):
        dest_base = self.dest_folder.get()
        method = self.organize_method.get()
        
        if method == "type":
            folder_name = self.get_file_category(file_path)
        elif method == "date":
            folder_name = self.get_date_folder(file_path)
        else:  # extension
            folder_name = self.get_extension_folder(file_path)
        
        if self.create_subfolders.get():
            dest_folder = os.path.join(dest_base, folder_name)
        else:
            dest_folder = dest_base
            
        return dest_folder
    
    def preview_organization(self):
        if not self.validate_inputs():
            return
        
        self.clear_log()
        self.log_message("Preview of file organization:")
        self.log_message("-" * 50)
        
        files = self.get_files_to_organize()
        if not files:
            self.log_message("No files found in source folder")
            return
        
        organization_plan = {}
        for file_path in files:
            dest_folder = self.get_destination_path(file_path)
            if dest_folder not in organization_plan:
                organization_plan[dest_folder] = []
            organization_plan[dest_folder].append(os.path.basename(file_path))
        
        for folder, files in organization_plan.items():
            self.log_message(f"\nFolder: {folder}")
            for file in files:
                self.log_message(f"  → {file}")
        
        self.log_message(f"\nTotal files to organize: {len(files)}")
        self.log_message(f"Total destination folders: {len(organization_plan)}")
    
    def validate_inputs(self):
        if not self.source_folder.get():
            messagebox.showerror("Error", "Please select a source folder")
            return False
        
        if not self.dest_folder.get():
            messagebox.showerror("Error", "Please select a destination folder")
            return False
        
        if not os.path.exists(self.source_folder.get()):
            messagebox.showerror("Error", "Source folder does not exist")
            return False
        
        if os.path.abspath(self.source_folder.get()) == os.path.abspath(self.dest_folder.get()):
            messagebox.showerror(
                "Error",
                "Destination folder must be different from source folder",
            )
            return False
        
        # Validate destination writability
        dest_path = self.dest_folder.get()
        try:
            # If destination exists, check if it's writable
            if os.path.exists(dest_path):
                if not os.access(dest_path, os.W_OK):
                    messagebox.showerror(
                        "Error",
                        f"Destination folder is not writable: {dest_path}"
                    )
                    return False
            elif os.access(os.path.dirname(dest_path) or ".", os.W_OK):
                # Destination missing → only probe parent dir; defer creation
                pass
                test_file = os.path.join(dest_path, ".write_test_temp")
                try:
                    with open(test_file, 'w') as f:
                        f.write("test")
                    os.remove(test_file)
                except (OSError, IOError):
                    messagebox.showerror(
                        "Error",
                        f"Cannot write to destination folder: {dest_path}"
                    )
                    return False
        except (OSError, IOError) as e:
            messagebox.showerror(
                "Error",
                f"Cannot access or create destination folder: {dest_path}\n{str(e)}"
            )
            return False
        
        return True
    
    def organize_files_thread(self, settings):
        """
        Worker thread that organizes files using captured settings dict.
        This avoids accessing Tkinter variables from non-main thread.
        """
        try:
            source = settings["source"]
            dest_base = settings["dest"]
            method = settings["method"]
            copy_files = settings["copy"]
            subfolders = settings["subfolders"]
            
            # Helper function to get destination path using captured settings
            def _get_destination_path(file_path):
                if method == "type":
                    folder_name = self.get_file_category(file_path)
                elif method == "date":
                    folder_name = self.get_date_folder(file_path)
                else:  # extension
                    folder_name = self.get_extension_folder(file_path)
                
                if subfolders:
                    dest_folder = os.path.join(dest_base, folder_name)
                else:
                    dest_folder = dest_base
                    
                return dest_folder
            
            # Get files to organize without using Tkinter variables
            if not source or not os.path.exists(source):
                files = []
            else:
                files = [
                    os.path.join(source, item)
                    for item in os.listdir(source)
                    if os.path.isfile(os.path.join(source, item))
                ]
            
            if not files:
                self._queue.put(lambda: self.log_message("No files found in source folder"))
                return
            
            self._queue.put(lambda: self.log_message(f"Starting organization of {len(files)} files..."))
            
            organized_count = 0
            error_count = 0
            
            for file_path in files:
                try:
                    dest_folder = _get_destination_path(file_path)
                    
                    # Create destination folder if it doesn't exist
                    os.makedirs(dest_folder, exist_ok=True)
                    
                    # Determine destination file path
                    filename = os.path.basename(file_path)
                    dest_path = os.path.join(dest_folder, filename)
                    
                    # Handle file name conflicts
                    counter = 1
                    while os.path.exists(dest_path):
                        name, ext = os.path.splitext(filename)
                        dest_path = os.path.join(dest_folder, f"{name}_{counter}{ext}")
                        counter += 1
                    
                    # Copy or move file
                    if copy_files:
                        shutil.copy2(file_path, dest_path)
                        action = "Copied"
                    else:
                        shutil.move(file_path, dest_path)
                        action = "Moved"
                    
                    self._queue.put(lambda f=filename, d=dest_folder, a=action: self.log_message(f"{a}: {f} → {d}"))
                    organized_count += 1
                    
                except Exception as e:
                    basename = os.path.basename(file_path)
                    error_msg = str(e)
                    self._queue.put(lambda f=basename, err=error_msg: self.log_message(f"Error processing {f}: {err}"))
                    error_count += 1
            
            self._queue.put(lambda: self.log_message("-" * 50))
            self._queue.put(lambda: self.log_message("Organization complete!"))
            self._queue.put(lambda c=organized_count: self.log_message(f"Files organized: {c}"))
            if error_count > 0:
                self._queue.put(lambda e=error_count: self.log_message(f"Errors encountered: {e}"))
            
        except Exception as e:
            self._queue.put(lambda err=str(e): self.log_message(f"Fatal error: {err}"))
        finally:
            self._queue.put(self.progress.stop)
            self._queue.put(lambda: self.status_label.config(text="Organization complete"))
    
    def organize_files(self):
        if not self.validate_inputs():
            return
        
        result = messagebox.askyesno("Confirm", "Are you sure you want to organize the files?")
        if not result:
            return
        
        self.clear_log()
        self.progress.start()
        self.status_label.config(text="Organizing files...")
        
        # Capture GUI state BEFORE spawning worker thread
        settings = {
            "source": self.source_folder.get(),
            "dest": self.dest_folder.get(),
            "method": self.organize_method.get(),
            "copy": self.copy_files.get(),
            "subfolders": self.create_subfolders.get(),
        }
        
        # Run organization in a separate thread to prevent GUI freezing
        thread = threading.Thread(
            target=self.organize_files_thread,
            args=(settings,),
        )
        thread.daemon = True
        thread.start()

def main():
    root = tk.Tk()
    app = FileOrganizer(root)
    root.mainloop()

if __name__ == "__main__":
    main()