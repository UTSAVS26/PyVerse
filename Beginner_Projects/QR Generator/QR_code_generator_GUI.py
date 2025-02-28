import qrcode
import customtkinter
from PIL import Image
import os
from tkinter import messagebox

class QRCodeGenerator:
    def __init__(self):
        self.app = customtkinter.CTk()
        self.app.title("QR Code Generator")
        self.app.geometry("350x600")
        
        self.img = None
        self.image_label = None
        
        self.setup_ui()
        
    def setup_ui(self):
        # Title Label
        label = customtkinter.CTkLabel(self.app, text="QR Code Generator", 
                                     font=("Consolas", 30))
        label.grid(row=0, column=0, padx=20, pady=20)
        
        # Entry Fields
        self.link_entry = customtkinter.CTkEntry(self.app, width=300)
        self.link_entry.grid(row=1, column=0, padx=20, pady=20)
        self.link_entry.configure(placeholder_text="Link")
        
        self.name_entry = customtkinter.CTkEntry(self.app, width=300)
        self.name_entry.grid(row=2, column=0, padx=20, pady=20)
        self.name_entry.configure(placeholder_text="Name of QR Code")
        
        # Buttons
        button = customtkinter.CTkButton(self.app, text="Generate QR Code", 
                                       command=self.generate_qr, width=300)
        button.grid(row=3, column=0, padx=20, pady=20)
        
        button_download = customtkinter.CTkButton(self.app, text="Download", 
                                                command=self.download_qr, width=300)
        button_download.grid(row=4, column=0, padx=20, pady=20)
        
        theme_button = customtkinter.CTkButton(self.app, text="Theme", 
                                             command=self.change_theme, width=300)
        theme_button.grid(row=7, column=0)
        
    def generate_qr(self):
        try:
            link = self.link_entry.get().strip()
            name = self.name_entry.get().strip()
            
            if not link or not name:
                messagebox.showerror("Error", "Please fill in both fields")
                return
                
            # Clean up previous image label if it exists
            if self.image_label:
                self.image_label.destroy()
                
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_M,
            )
            qr.add_data(link)
            qr.make(fit=True)
            
            self.img = qr.make_image(fill_color="black", back_color="white")
            temp_path = f"temp_{name}.png"
            self.img.save(temp_path)
            
            my_image = customtkinter.CTkImage(
                light_image=Image.open(temp_path),
                dark_image=Image.open(temp_path),
                size=(200, 200)
            )
            
            self.image_label = customtkinter.CTkLabel(self.app, image=my_image, text="")
            self.image_label.grid(row=5, column=0)
            
            # Clean up temporary file
            os.remove(temp_path)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate QR code: {str(e)}")
            
    def download_qr(self):
        try:
            if not self.img:
                messagebox.showerror("Error", "Please generate a QR code first")
                return
                
            name = self.name_entry.get().strip()
            if not name:
                messagebox.showerror("Error", "Please enter a name for the file")
                return
                
            file_path = f"{name}.png"
            if os.path.exists(file_path):
                if not messagebox.askyesno("Warning", "File already exists. Overwrite?"):
                    return
                    
            self.img.save(file_path)
            messagebox.showinfo("Success", f"QR code saved as {file_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save QR code: {str(e)}")
            
    def change_theme(self):
        current_theme = customtkinter.get_appearance_mode()
        new_theme = "light" if current_theme == "Dark" else "dark"
        customtkinter.set_appearance_mode(new_theme)
        
    def run(self):
        self.app.mainloop()

if __name__ == "__main__":
    app = QRCodeGenerator()
    app.run()