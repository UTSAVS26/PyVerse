import os
import re
import tkinter as tk
from threading import Thread
from itertools import count
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import pandas as pd


def excel_to_csv(input_file, output_folder):
    try:
        xls = pd.read_excel(input_file, sheet_name=None)
        os.makedirs(output_folder, exist_ok=True)
        results = []

        total_sheets = len(xls)
        current_sheet = 0

        for idx, (sheet_name, df) in enumerate(xls.items(), start=1):
            safe_title = re.sub(r'[^A-Za-z0-9._-]', "_", sheet_name)
            if not safe_title:
                safe_title = f"Sheet{idx}"

            base_name = f"{Path(input_file).stem}_{safe_title}"
            for i in count():
                suffix = f"_{i}" if i > 0 else ""
                candidate = os.path.join(output_folder, f"{base_name}{suffix}.csv")
                if not os.path.exists(candidate):
                    break

            df = df.where(pd.notnull(df), "")
            df.to_csv(candidate, index=False, encoding='utf-8')

            current_sheet += 1
            progress_percent = (current_sheet / total_sheets) * 100
            yield progress_percent

            results.append(candidate)

        return True, f"Successfully converted {len(results)} sheet(s) to CSV."
    except FileNotFoundError:
        return False, "Error: Input file not found."
    except PermissionError:
        return False, "Error: File is open or access is denied."
    except Exception as e:
        return False, f"Error: {str(e)}"


class ExcelToCSVApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Excel to CSV Converter")
        self.root.geometry("500x500")
        self.root.configure(bg="#252526")
        self.root.minsize(600, 400)

        self.style = ttk.Style()
        self.style.theme_use('clam')

        # Define default and hover styles
        self.style.configure("Primary.TButton", background="#007acc", foreground="#ffffff", font=("Arial", 11, "bold"))
        self.style.map("Primary.TButton", background=[('active', '#005f99')])

        self.style.configure("Hover.TButton", background="#1a8cff", foreground="#ffffff", font=("Arial", 11, "bold"))

        self.style.configure("Secondary.TButton", background="#505050", foreground="#ffffff")
        self.style.map("Secondary.TButton", background=[('active', '#606060')])

        self.style.configure("TLabel", font=("Arial", 12), background="#252526", foreground="#d4d4d4")
        self.style.configure("TEntry", padding=10, font=("Arial", 11), fieldbackground="#3c3c3c", foreground="#ffffff")
        self.style.map("TEntry", fieldbackground=[('disabled', '#3c3c3c')], selectbackground=[('focus', '#007acc')])
        self.style.configure("TProgressbar", thickness=4, troughcolor="#3c3c3c", background="#007acc")
        self.style.configure("Card.TFrame", background="#2d2d2d", relief="flat")

        self.main_frame = ttk.Frame(self.root, style="Card.TFrame")
        self.main_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.header_frame = ttk.Frame(self.main_frame, style="Card.TFrame")
        self.header_frame.grid(row=0, column=0, sticky="ew", pady=(0, 20))
        self.main_frame.columnconfigure(0, weight=1)

        ttk.Label(
            self.header_frame,
            text="Excel to CSV Converter",
            font=("Arial", 20, "bold"),
            foreground="#ffffff",
            background="#2d2d2d"
        ).pack(pady=10)

        self.input_frame = ttk.Frame(self.main_frame, style="Card.TFrame")
        self.input_frame.grid(row=1, column=0, sticky="ew", padx=50)
        self.input_frame.columnconfigure(0, weight=1)

        ttk.Label(self.input_frame, text="Excel File", font=("Arial", 12, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 8))

        self.input_entry = ttk.Entry(self.input_frame)
        self.input_entry.grid(row=1, column=0, sticky="ew", padx=(0, 10))

        ttk.Button(self.input_frame, text="Browse", command=self.browse_input, style="Secondary.TButton").grid(row=1, column=1, padx=5)

        ttk.Label(self.input_frame, text="Output Folder", font=("Arial", 12, "bold")).grid(row=2, column=0, sticky="w", pady=(20, 8))

        self.output_entry = ttk.Entry(self.input_frame)
        self.output_entry.grid(row=3, column=0, sticky="ew", padx=(0, 10))

        ttk.Button(self.input_frame, text="Browse", command=self.browse_output, style="Secondary.TButton").grid(row=3, column=1, padx=5)

        self.convert_btn = ttk.Button(self.input_frame, text="Convert to CSV", command=self.start_conversion, style="Primary.TButton")
        self.convert_btn.grid(row=4, column=0, columnspan=2, pady=30, sticky="ew")

        # Hover effect using style switch
        self.convert_btn.bind("<Enter>", lambda e: self.convert_btn.configure(style="Hover.TButton"))
        self.convert_btn.bind("<Leave>", lambda e: self.convert_btn.configure(style="Primary.TButton"))

        self.progress = ttk.Progressbar(self.input_frame, mode='indeterminate')
        self.progress.grid(row=5, column=0, columnspan=2, pady=15, sticky="ew")

        self.status_label = ttk.Label(self.input_frame, text="Ready", foreground="#007acc")
        self.status_label.grid(row=6, column=0, columnspan=2, pady=5)

        self.root.bind("<Configure>", self.on_resize)

    def on_resize(self, event):
        width = max(600, event.width)
        entry_width = int(width // 20)
        progress_length = int(width * 0.6)
        self.input_entry.configure(width=entry_width)
        self.output_entry.configure(width=entry_width)
        self.progress.configure(length=progress_length)

    def browse_input(self):
        file = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx *.xls")])
        if file:
            self.input_entry.delete(0, tk.END)
            self.input_entry.insert(0, file)

    def browse_output(self):
        folder = filedialog.askdirectory()
        if folder:
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, folder)

    def start_conversion(self):
        input_file = self.input_entry.get()
        output_folder = self.output_entry.get()

        if not input_file or not output_folder:
            messagebox.showerror("Error", "Please select both input file and output folder.")
            return

        self.convert_btn.config(state='disabled')
        self.progress.configure(mode='determinate')
        self.progress.start()

        def convert():
            try:
                for progress_percent in excel_to_csv(input_file, output_folder):
                    self.progress['value'] = progress_percent
                    self.progress.update_idletasks()
                self.finish_conversion(True, f"Successfully converted all sheets to CSV in {output_folder}.")
            except Exception as e:
                self.finish_conversion(False, f"Error: {str(e)}")

        Thread(target=convert, daemon=True).start()

    def finish_conversion(self, success, message):
        self.progress.stop()
        self.progress.configure(mode='indeterminate')
        self.convert_btn.config(state='normal')
        self.status_label.config(text=message, foreground="#007acc" if success else "#ff5555")
        if success:
            messagebox.showinfo("Success", message)
        else:
            messagebox.showerror("Error", message)


if __name__ == "__main__":
    root = tk.Tk()
    app = ExcelToCSVApp(root)
    root.mainloop()
