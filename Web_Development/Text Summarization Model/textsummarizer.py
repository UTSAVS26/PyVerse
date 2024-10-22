import streamlit as st
from transformers import pipeline

# Title of the web app
st.title("Text Summarization Tool")

# Load the summarization model
@st.cache_resource(show_spinner=True)  # Cache the model loading for faster performance
def load_summarizer():
    return pipeline("summarization", model="t5-small")

summarizer = load_summarizer()

# Instructions for users
st.write("Enter the text you'd like to summarize (minimum 50 words).")

# Create a text area for the user to input text
user_input = st.text_area("Input Text", height=200)

# A button to initiate the summarization process
if st.button("Summarize"):
    if len(user_input.split()) < 50:
        st.warning("Please enter at least 50 words for summarization.")
    else:
        # Show a spinner while the summarization is being processed
        with st.spinner("Summarizing..."):
            # Generate the summary
            summary = summarizer(user_input, max_length=150, min_length=30, do_sample=False)
            # Display the summarized text
            st.subheader("Summary:")
            st.write(summary[0]['summary_text'])
