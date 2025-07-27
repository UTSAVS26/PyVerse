%%writefile app.py
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

# Load and train on built-in sample dataset
df=pd.read_csv("/content/synthetic_sms_data.csv")  # Make sure this file is in the same folder

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['spam'], test_size=0.2, random_state=42)

# Build pipeline: TF-IDF + Logistic Regression
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('model', LogisticRegression(max_iter=1000))
])

# Train the model
pipeline.fit(X_train, y_train)

# Streamlit app
st.title("📬 SPAM SHIELD-Spam Email Predictor")

st.subheader("Paste your email text below:")
user_input = st.text_area("Email Text", height=200)

if st.button("Check if Spam"):
    if user_input.strip() == "":
        st.warning("Please enter some email text.")
    else:
        prediction = pipeline.predict([user_input])[0]
        probability = pipeline.predict_proba([user_input])[0][1]

        if prediction == 1:
            st.error(f"🔴 This email is **Spam** (Confidence: {probability:.2%})")
        else:
            st.success(f"🟢 This email is **Not Spam** (Confidence: {(1 - probability):.2%})")
