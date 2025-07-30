import streamlit as st
import joblib
import pandas as pd

#Load pre-trained model
import os
MODEL_PATH = os.path.join(os.path.dirname(__file__), "Job_Internship_model.pkl")
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    st.error("Model file not found. Please ensure the model is trained and saved.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()


st.title("🎯Internship Fraud Detection🎯")

st.subheader("Predicting Internship is Scam or Genuine")

st.write("Enter the details below to check if the internship is fraud or genuine:")

Title=st.text_input("Enter Title Of Internship",placeholder="e.g., Software Engineering Intern at Google")
Description=st.text_input("Enter Description of Internship",placeholder="Paste the full job description here...")

if st.button("Predict"):
    if Title and Description:

        Input_data=pd.DataFrame({
            "Title":[Title],
            "Description":[Description]
        })

        prediction = model.predict(Input_data)[0]
        prediction_proba = model.predict_proba(Input_data)[0]
        
        if prediction == 1:
           confidence = prediction_proba[1] * 100
           st.error(f"⚠️ This Internship is likely a SCAM ❌")
           st.warning(f"Confidence: {confidence:.1f}%")
        else:
            st.success(f"✅ This Internship appears to be GENUINE")
            st.info(f"Confidence: {prediction_proba[0] * 100:.1f}%")