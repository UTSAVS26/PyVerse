import pdf_data
from pathlib import Path
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage


images_folder = Path(__file__).parent

def file2_app(window,file_path):
    OUTPUT_PATH = Path(__file__).parent
    ASSETS_PATH = OUTPUT_PATH / Path(images_folder/'build2/assets/frame0')

    def pdf_to_data(file_path):
        global image
        image , n  = pdf_data.open_file(file_path)
        print(n)
        return n


    def relative_to_assets(path: str) -> Path:
        return ASSETS_PATH / Path(path)
    
    
    window.geometry("771x538")
    window.configure(bg = "#F7F7F7")


    canvas = Canvas(
        window,
        bg = "#F7F7F7",
        height = 538,
        width = 771,
        bd = 0,
        highlightthickness = 0,
        relief = "ridge"
    )

    canvas.place(x = 0, y = 0)
    canvas.create_rectangle(
        0.0,
        2.0,
        771.0,
        59.0,
        fill="#D9D9D9",
        outline="")

    canvas.create_text(
        10.0,
        9.0,
        anchor="nw",
        text="Mediextract",
        fill="#000000",
        font=("Inter Thin", 24 * -1)
    )

    canvas.create_rectangle(
        16.0,
        83.0,
        365.0,
        489.0,
        fill="#D9D9D9",
        outline="")

    button_image_1 = PhotoImage(
        file=relative_to_assets("button_1.png"))
    button_1 = Button(
        image=button_image_1,
        borderwidth=0,
        highlightthickness=0,
        command=lambda: pdf_to_data(file_path),
        relief="flat"
    )
    button_1.place(
        x=391.0,
        y=467.0,
        width=168.0,
        height=44.0
    )

    button_image_2 = PhotoImage(
        file=relative_to_assets("button_2.png"))
    button_2 = Button(
        image=button_image_2,
        borderwidth=0,
        highlightthickness=0,
        command=lambda:pdf_to_data(file_path),
        relief="flat"
    )
    button_2.place(
        x=574.0,
        y=467.0,
        width=163.0,
        height=44.0
    )

    entry_image_1 = PhotoImage(
        file=relative_to_assets("entry_1.png"))
    entry_bg_1 = canvas.create_image(
        640.0,
        151.0,
        image=entry_image_1
    )
    entry_1 = Entry(
        bd=0,
        bg="#A8A8A8",
        fg="#000716",
        highlightthickness=0
    )
    entry_1.place(
        x=550.0,
        y=136.0,
        width=180.0,
        height=28.0
    )

    entry_image_2 = PhotoImage(
        file=relative_to_assets("entry_2.png"))
    entry_bg_2 = canvas.create_image(
        640.0,
        208.0,
        image=entry_image_2
    )
    entry_2 = Entry(
        bd=0,
        bg="#A8A8A8",
        fg="#000716",
        highlightthickness=0
    )
    entry_2.place(
        x=550.0,
        y=193.0,
        width=180.0,
        height=28.0
    )

    entry_image_3 = PhotoImage(
        file=relative_to_assets("entry_3.png"))
    entry_bg_3 = canvas.create_image(
        640.5,
        330.5,
        image=entry_image_3
    )
    entry_3 = Entry(
        bd=0,
        bg="#A8A8A8",
        fg="#000716",
        highlightthickness=0
    )
    entry_3.place(
        x=551.0,
        y=254.0,
        width=179.0,
        height=151.0
    )

    canvas.create_text(
        404.0,
        132.0,
        anchor="nw",
        text="Name:",
        fill="#000000",
        font=("Inter", 20 * -1)
    )

    canvas.create_text(
        404.0,
        193.0,
        anchor="nw",
        text="Age:",
        fill="#000000",
        font=("Inter", 20 * -1)
    )

    canvas.create_text(
        404.0,
        261.0,
        anchor="nw",
        text="Prescription:",
        fill="#000000",
        font=("Inter Medium", 20 * -1)
    )
    window.resizable(False, False)
    
    window.mainloop()