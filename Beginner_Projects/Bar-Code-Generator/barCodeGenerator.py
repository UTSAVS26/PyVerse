from barcode.writer import ImageWriter
from barcode import EAN13, EAN8, UPCA
import streamlit as st
import base64
import os
from io import BytesIO

# Dictionary to map barcode types to their respective lengths and classes
BARCODE_TYPE = {"EAN-13": [13, EAN13], "EAN-8": [8, EAN8], "UPCA": [12, UPCA]}

def barCodeGenerator():
    """
    Main function to render the Streamlit interface for barcode generation.
    """
    # Display the title of the app
    st.markdown(
        """
        <h1 style='text-align: center; color: #FF4B4B; font-family: Verdana;'>
            BAR CODE GENERATOR📊
        </h1>
        """,
        unsafe_allow_html=True,
    )

    # Create a container to hold all the elements
    box = st.container()
    with box:
        # Radio buttons to select the type of barcode
        option = st.radio(
            "Select type of Barcode", ["EAN-13", "EAN-8", "UPCA"], horizontal=True
        )
        # Text input to enter the barcode number
        num = st.text_input(
            "Enter barcode number",
            value="",
            max_chars=BARCODE_TYPE[option][0],
            placeholder=f"Enter {BARCODE_TYPE[option][0]} digits long barcode number",
        )
        # Placeholder for the button
        button_div = st.empty()

    with button_div:
        # Button to generate the barcode
        if st.button("Generate barcode"):
            generate(num, box, option)
            # Button to reset the barcode
            st.button("Reset barcode")

def generate(num, box, option):
    """
    Function to generate and display the barcode.

    Parameters:
    num (str): The barcode number entered by the user.
    box (streamlit.container): The container to hold the elements.
    option (str): The type of barcode selected by the user.
    """
    with box:
        # Validate the barcode number
        if len(num) != BARCODE_TYPE[option][0] or not num.isnumeric():
            st.warning(
                f"Please enter a valid {option} barcode of {BARCODE_TYPE[option][0]} digits!!"
            )
        else:
            # Generate the barcode and save it to a BytesIO stream
            my_code = BARCODE_TYPE[option][1](num, writer=ImageWriter())
            image_stream = BytesIO()
            my_code.write(image_stream)
            image_stream.seek(0)

            # Encode the image for display
            encoded_image = base64.b64encode(image_stream.read()).decode()

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

# Run the main function to start the Streamlit app
barCodeGenerator()