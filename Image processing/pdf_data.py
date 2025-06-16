# from pdf2image import convert_from_path
import tkinter as tk
from tkinter import filedialog
import pytesseract
import numpy as np
import cv2
from PIL import Image
import re


def open_file(file_path,V):
    # global count 
    # count = V
    if file_path:
        pages = convert_from_path(file_path, poppler_path='C:\\Users\\ACER\\Downloads\\Release-24.08.0-0\\poppler-24.08.0\\Library\\bin')
        

        def preprocess_image(img):
            gray = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
            processed_image = cv2.adaptiveThreshold(
                resized,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 
                61,
                11
            )
            return processed_image
        
        img = preprocess_image(pages[V])
       
        
        


        pytesseract.pytesseract.tesseract_cmd=r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
        text = pytesseract.image_to_string(img, lang='eng')
        # pytesseract.pytesseract.tesseract_cmd=r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
        # text2 = pytesseract.image_to_string(img2, lang='eng')
        # print(text1)
        # print(text2)
        

        pattern = 'Name:(.*)Date'
        matches1 = re.findall(pattern, text)

        pattern = 'Address:(.*)\n'
        matches2 = re.findall(pattern, text)

        pattern = 'Address[^\n]*(.*)Directions'
        matches3 = re.findall(pattern, text, flags=re.DOTALL)
        
        


        
        return matches1[0].strip().strip('_') , matches2[0].strip().strip('_'), matches3[0].strip().strip('_')