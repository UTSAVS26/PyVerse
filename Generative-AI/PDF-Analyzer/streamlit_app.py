import streamlit as st
import requests

FLASK_API_URL = "http://localhost:5000/generate"

st.title("PDF Q&A App")
st.write("Upload a PDF and ask any question related to its content.")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

question = st.text_input("Enter your question about the PDF:")

if st.button("Ask Question"):
    if uploaded_file and question:
        files = {"fileUploaded": uploaded_file}
        data = {"fileDoubt": question}
        
        response = requests.post(FLASK_API_URL, files=files, data=data)
        
        if response.status_code == 200:
            response_data = response.json()
            st.write("### Question:")
            st.write(response_data.get("prompt", ""))
            st.write("### Answer:")
            st.write(response_data.get("response", ""))
        else:
            st.error("Error: Could not retrieve the response from the server.")
    else:
        st.warning("Please upload a PDF and enter a question.")
