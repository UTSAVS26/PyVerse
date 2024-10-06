import qrcode
import customtkinter
from PIL import Image  
from customtkinter import *

app = customtkinter.CTk()
app.title("QR Code Generator")
app.geometry("350x600")

img = None
link = None
qr_image = None
image_label = None

def generate_qr():

    global link, qr_image, image_label, img
    
    link = link_entry.get()
    name = name_entry.get()

    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
    )
    qr.add_data(link)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    img.save(f"{name}.png")
    
    my_image = customtkinter.CTkImage(light_image=Image.open(f"{name}.png"),
                                  dark_image=Image.open(f"{name}.png"),
                                  size=(200, 200))


    image_label = customtkinter.CTkLabel(app, image=my_image, text="") 
    image_label.grid(row=5, column=0)

def download_qr():
    global link, qr_image, image_label

    file = f"{name_entry.get()}.png"
    img.save(file)


label = customtkinter.CTkLabel(app, text="QR Code Generator", font=("Consolas", 30), compound="center")
label.grid(row=0, column=0, padx=20, pady=20)

link_entry = customtkinter.CTkEntry(app, width=300)
link_entry.grid(row=1, column=0, padx=20, pady=20)

link_entry.configure(placeholder_text="Link")

name_entry = customtkinter.CTkEntry(app, width=300)
name_entry.grid(row=2, column=0, padx=20, pady=20)

name_entry.configure(placeholder_text="Name OF QR Code")

button = customtkinter.CTkButton(app, text="Generate QR Code", command=generate_qr, width=300)
button.grid(row=3, column=0, padx=20, pady=20)

button_download = CTkButton(app, text="Download", command=download_qr, width=300)
button_download.grid(row=4, column=0, padx=20, pady=20)

image_label = None

def change_theme():
  if customtkinter.get_appearance_mode() == "Light":
    customtkinter.set_appearance_mode("dark")
  else:
    customtkinter.set_appearance_mode("light")

# Theme Change Btton
theme_button = customtkinter.CTkButton(app, text="Theme", command=change_theme, width=300)
theme_button.grid(row=7, column=0)

app.mainloop()