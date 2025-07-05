import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("waste_classifier_model.keras")

# Function to make predictions
def predict_fun(img):
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (96, 96))  # match MobileNetV2 input size
    img = img / 255.0  # normalize as done in training
    img = np.reshape(img, [-1, 96, 96, 3])

    result = model.predict(img)[0][0]

    if result < 0.5:
        return 'The Image Shown is Organic Waste'      # class 0
    else:
        return 'The Image Shown is Recyclable Waste'   # class 1


# Streamlit UI with custom CSS for center alignment
st.set_page_config(page_title="Waste Classification App", layout="wide")

# Add custom CSS
st.markdown("""
    <style>
        .main > div {
            max-width: 800px;
            padding-left: 50px;
            padding-right: 50px;
            margin: 0 auto;
        }
        .stMarkdown {
            text-align: center;
        }
        .stButton > button {
            margin: 0 auto;
            display: block;
        }
        .stTitle {
            text-align: center;
        }
        div[data-testid="stFileUploader"] {
            text-align: center;
        }
        .app-icon {
            display: block;
            margin: 0 auto;
            width: 175px;
            height: 175px;
            padding: 10px;
        }
        .info-text {
            font-size: 1.2rem;
            line-height: 1.6;
            margin-bottom: 20px;
        }
        .button-spacing {
            margin-top: 40px;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state for navigation
if 'page' not in st.session_state:
    st.session_state.page = 'Home'

if st.session_state.page == "Home":
    # Social links container in top right
    social_container = st.container()
    with social_container:
        col1, col2, col3, col4 = st.columns([6, 1, 1, 1])
        with col2:
            st.markdown("[LinkedIn](https://www.linkedin.com/in/aditya-kumar-3721012aa)")
        with col3:
            st.markdown("[Twitter](https://x.com/kaditya264?s=09)")
        with col4:
            st.markdown("[GitHub](https://github.com/GxAditya)")
    
    # Add app icon below social links
    st.markdown("""
        <div style='text-align: center;'>
            <img src='https://raw.githubusercontent.com/GxAditya/Waste-Classification/main/waste.png' class='app-icon'>
        </div>
    """, unsafe_allow_html=True)
    
    # Main content
    st.markdown("<h1 style='text-align: center;'>Waste Classification Using Deep Learning</h1>", unsafe_allow_html=True)
    st.markdown("<p class='info-text' style='text-align: center;'>This project uses a Convolutional Neural Network (CNN) to classify waste as either Organic or Recyclable.</p>", unsafe_allow_html=True)
    st.markdown("<p class='info-text' style='text-align: center;'>The model is trained on a dataset of labeled images to improve environmental waste management.</p>", unsafe_allow_html=True)
    
    st.markdown("<h3 style='text-align: center;'>How it Works</h3>", unsafe_allow_html=True)
    st.markdown("""
        <div class='info-text' style='text-align: center;'>
        1. Upload an image of waste material.<br>
        2. The model analyzes the image and classifies it as Organic or Recyclable.<br>
        3. This helps in proper disposal and recycling efforts.
        </div>
        <div class='button-spacing'></div>
    """, unsafe_allow_html=True)
    
    # Center the "Try It Now" button
    _, col2, _ = st.columns([1, 2, 1])
    with col2:
        if st.button("Try It Now!", use_container_width=True):
            st.session_state.page = "Classification"
            st.rerun()

elif st.session_state.page == "Classification":
    st.markdown("<h1 style='text-align: center;'>Waste Classification</h1>", unsafe_allow_html=True)
    
    # Back button centered
    _, col2, _ = st.columns([1, 2, 1])
    with col2:
        if st.button("← Back to Home", use_container_width=True):
            st.session_state.page = "Home"
            st.rerun()
    
    st.markdown("<p style='text-align: center;'>Upload an image to classify it as Organic or Recyclable.</p>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Center the image and prediction
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            if st.button("Classify Image", use_container_width=True):
                result = predict_fun(image)
                st.markdown(f"<h3 style='text-align: center;'>Prediction:</h3>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center;'>{result}</p>", unsafe_allow_html=True)
