from pathlib import Path
import gui_data
import vedio_to_pdf
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage
from tkinter import filedialog
from pdf2image import convert_from_path
import tkinter as tk
from tkinter import filedialog
import pytesseract
import numpy as np
import cv2
from PIL import Image
import re

images_folder1 = Path(__file__).parent / 'build/assets/frame0'
images_folder2 = Path(__file__).parent / 'build2/assets/frame0'

vedio_to_data_button = images_folder1 / 'button_1.png'
pdf_to_data_button = images_folder1 / 'button_2.png'
tempo_button = images_folder1 / 'button_3.png'
white_1_button = images_folder2 / 'entry_1.png'
white_2_button = images_folder2 / 'entry_2.png'
white_3_button = images_folder2 / 'entry_3.png'








global count 

def video_to_data(window):
    video_path = filedialog.askopenfilename()
    if video_path:
        output_pdf= images_folder1 / 'vedio_to_image.pdf'
        temprorary_folder = images_folder1 / 'temproraryfolder'
        vedio_to_pdf.vedio_to_pdf(video_path, output_pdf, temprorary_folder)
        pdf_to_data(window, file_path=output_pdf)

def pdf_to_data(window, file_path="None"):
    if file_path == "None":
        file_path = filedialog.askopenfilename()
    if file_path:
        pages = convert_from_path(file_path, poppler_path='C:\\Users\\ACER\\Downloads\\Release-24.08.0-0\\poppler-24.08.0\\Library\\bin')
        gui_data.file1_app(window, file_path, pages)

def main():
    OUTPUT_PATH = Path(__file__).parent
    ASSETS_PATH = OUTPUT_PATH / Path(images_folder1 / 'build/assets')

    # def relative_to_assets(path: str) -> Path:
    #     return ASSETS_PATH / Path(path)

    window = Tk()
    window.geometry("970x628")
    window.configure(bg = "#D0EEF2")

    canvas = Canvas(
        window,
        bg = "#D0EEF2",
        height = 628,
        width = 970,
        bd = 0,
        highlightthickness = 0,
        relief = "ridge"
    )

    canvas.place(x = 0, y = 0)
    canvas.create_rectangle(
        0.0,
        0.0,
        970.0,
        85.0,
        fill="#000026",
        outline=""
    )

    # Video to Data button
    button_image_1 = PhotoImage(
        file=vedio_to_data_button
    )
    button_1 = Button(
        image=button_image_1,
        borderwidth=0,
        highlightthickness=0,
        command=lambda: video_to_data(window),
        relief="flat"
    )
    button_1.place(
        x=76.0,
        y=444.0,
        width=230.0,
        height=62.0
    )

    # PDF to Data button
    button_image_2 = PhotoImage(
        file=pdf_to_data_button
    )
    button_2 = Button(
        image=button_image_2,
        borderwidth=0,
        highlightthickness=0,
        command=lambda: pdf_to_data(window),
        relief="flat"
    )
    button_2.place(
        x=80.0,
        y=538.0,
        width=230.0,
        height=55.0
    )

    button_image_3 = PhotoImage(
        file=tempo_button
    )
    button_3 = Button(
        image=button_image_3,
        borderwidth=0,
        highlightthickness=0,
        relief="flat"
    )
    button_3.place(
        x=60.0,
        y=107.0,
        width=846.0,
        height=284.0
    )

    window.resizable(False, False)
    window.mainloop()

if __name__ == "__main__":
    main()
