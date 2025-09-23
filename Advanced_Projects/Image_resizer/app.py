import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image


class ImageResizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Resizer")
        self.root.geometry("500x350")
        self.root.configure(bg="#2b2b2b")

        self.images = []
        self.output_folder = os.path.join(os.getcwd(), "output")

        self._build_ui()

    def _build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TButton", background="#444444", foreground="white")
        style.configure("TLabel", background="#2b2b2b", foreground="white")

        # Select Images
        ttk.Button(self.root, text="📂 Select Images", command=self._select_images).pack(pady=10)

        # Width/Height
        frame1 = tk.Frame(self.root, bg="#2b2b2b")
        frame1.pack(pady=5)
        tk.Label(frame1, text="Width:", bg="#2b2b2b", fg="white").grid(row=0, column=0, padx=5)
        self.width_entry = tk.Entry(frame1, bg="#3c3f41", fg="white")
        self.width_entry.grid(row=0, column=1, padx=5)

        tk.Label(frame1, text="Height:", bg="#2b2b2b", fg="white").grid(row=0, column=2, padx=5)
        self.height_entry = tk.Entry(frame1, bg="#3c3f41", fg="white")
        self.height_entry.grid(row=0, column=3, padx=5)

        # Percentage
        frame2 = tk.Frame(self.root, bg="#2b2b2b")
        frame2.pack(pady=5)
        tk.Label(frame2, text="Scale (%):", bg="#2b2b2b", fg="white").grid(row=0, column=0, padx=5)
        self.scale_entry = tk.Entry(frame2, bg="#3c3f41", fg="white")
        self.scale_entry.grid(row=0, column=1, padx=5)

        # Output folder
        ttk.Button(self.root, text="📁 Select Output Folder", command=self._select_output_folder).pack(pady=10)

        # Resize button
        ttk.Button(self.root, text="⚡ Resize", command=self._resize_images).pack(pady=10)

    def _select_images(self):
        files = filedialog.askopenfilenames(
            title="Select Images",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        if files:
            self.images = files
            messagebox.showinfo("Selected", f"{len(files)} image(s) selected.")

    def _select_output_folder(self):
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.output_folder = folder
            messagebox.showinfo("Output Folder", f"Output folder set to:\n{folder}")

    def _resize_images(self):
        if not self.images:
            messagebox.showerror("Error", "No images selected.")
            return

        width = self.width_entry.get()
        height = self.height_entry.get()
        scale = self.scale_entry.get()

        try:
            width = int(width) if width else None
            height = int(height) if height else None
            scale = int(scale) if scale else None

            for img_path in self.images:
                self._resize_single(img_path, width, height, scale)

            messagebox.showinfo("Success", "Images resized successfully!")

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _resize_single(self, input_path, width, height, scale):
        img = Image.open(input_path)
        orig_width, orig_height = img.size

        if scale:
            width = int(orig_width * scale / 100)
            height = int(orig_height * scale / 100)
        elif width and height:
            width, height = int(width), int(height)
        else:
            raise ValueError("Provide width/height or scale.")

        resized = img.resize((width, height), Image.Resampling.LANCZOS)

        os.makedirs(self.output_folder, exist_ok=True)
        base, ext = os.path.splitext(os.path.basename(input_path))
        output_file = os.path.join(self.output_folder, f"{base}_resized{ext}")
        resized.save(output_file)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageResizerApp(root)
    app.run()
