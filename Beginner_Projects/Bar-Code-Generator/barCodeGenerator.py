from barcode.writer import ImageWriter
from barcode import EAN13, EAN8, UPCA
import streamlit as st
import base64
import os

# Set barcode type
BARCODE_TYPE = {"EAN-13": [13, EAN13], "EAN-8": [8, EAN8], "UPCA": [12, UPCA]}

def barCodeGenerator():
    st.markdown(
    """
    <h1 style='text-align: center; color: #FF4B4B; font-family: Verdana;'>
        BAR CODE GENERATOR📊
    </h1>
    """,
    unsafe_allow_html=True,
    )

    box = st.container()  # To keep everything inside one container
    with box:
        option = st.radio(
            "Select type of Barcode", ["EAN-13", "EAN-8", "UPCA"], horizontal=True
        )
        num = st.text_input(
            "Enter barcode number",
            value="",
            max_chars=BARCODE_TYPE[option][0],
            placeholder=f"Enter {BARCODE_TYPE[option][0]} digits long barcode number",
        )
        button_div = st.empty()  # So that when Generate Barcode is pressed, it will be replaced by Reset button
    
    with button_div:
        if st.button("Generate barcode"): 
            generate(num, box, option)
            st.button("Reset barcode")  # Resets everything


def generate(num, box, option):
    with box:
        if len(num) != BARCODE_TYPE[option][0] or not num.isnumeric():
            st.warning(
                f"Please enter a valid {option} barcode of {BARCODE_TYPE[option][0]} digits!!"
            )
        else:
            # Path for the image
            image_path = "assets/barcode"

            # Create the 'assets' directory if it doesn't exist
            if not os.path.exists('assets'):
                os.makedirs('assets')

            # Generate the barcode and save it
            my_code = BARCODE_TYPE[option][1](num, writer=ImageWriter())
            my_code.save(image_path)

            # Open the saved image and encode it for display
            with open(f"{image_path}.png", "rb") as file:
                image_data = file.read()

            encoded_image = base64.b64encode(image_data).decode()

            # Display the barcode image and download button
            st.markdown(
                f"""
                <style>
                .button-container {{
                    display: flex;
                    justify-content: space-around;
                }}
                .styled-button {{
                    background-color: #ff4b4b;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 5px;
                    cursor: pointer;
                    transition: background-color 0.3s, color 0.3s;
                }}
                .styled-button:hover {{
                    background-color: #ff3333;
                    color: white;
                }}
                </style>
                <div style="display: flex; flex-direction: column; align-items: center;">
                    <img src="data:image/png;base64,{encoded_image}" alt="Your Image">
                    <br>
                    <a href="data:image/png;base64,{encoded_image}" download="barcode.png">
                        <button class="styled-button">Download Image</button>
                    </a>
                </div>
            """,
                unsafe_allow_html=True,
            )

barCodeGenerator()
