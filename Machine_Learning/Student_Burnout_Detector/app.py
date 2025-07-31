# app.py


import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load('burnout_model_dt.joblib')


st.set_page_config(page_title="🧠 Burnout Risk Detector")
st.title("🧠 Student Burnout Risk Detector")

st.write("Fill your daily data below:")

# Input form (all as integers, matching training data)
sleep = st.slider("Sleep duration (hours)", 0, 12, 6)
screen = st.slider("Screen time (hours)", 0, 15, 6)
activity = st.slider("Physical activity (hours)", 0, 5, 1)
mood = st.slider("Mood level (1=low, 5=high)", 1, 5, 3)
assignments = st.slider("Number of assignments", 0, 10, 2)
caffeine = st.slider("Caffeine intake (cups)", 0, 5, 1)
social = st.slider("Social interaction (hours)", 0, 10, 2)

if st.button("🔍 Predict Burnout Risk"):
    # Match feature order used during training
    user_data = np.array([[sleep, screen, activity, mood, assignments, caffeine, social]])
    prediction = model.predict(user_data)[0]

    if prediction == 0:
        st.success("✅ You seem healthy! Keep it up!")
    elif prediction == 1:
        st.warning("⚠️ Mild burnout risk. Consider better balance & self-care.")
    else:
        st.error("🚨 High burnout risk! Please take care & seek help if needed.")

    st.subheader("📊 Your inputs:")
    st.write(f"Sleep: {sleep} hrs, Screen: {screen} hrs, Activity: {activity} hrs, "
             f"Mood: {mood}, Assignments: {assignments}, Caffeine: {caffeine} cups, "
             f"Social: {social} hrs")
