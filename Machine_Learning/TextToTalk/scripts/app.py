def main():
    """Streamlit app for PDF to MP3 conversion."""
    st.title("PDF to MP3 Converter")

    uploaded_file = st.file_uploader("Choose a PDF file to convert:", type=['pdf'])

    if uploaded_file is not None:
        audio_file, error_message = pipeline(uploaded_file, lang='en')  # Pass the selected language here
        if audio_file:
            st.audio(audio_file, format='audio/mp3')
            st.download_button(
                label="Download Audio",
                data=audio_file,
                file_name="output.mp3",
                mime="audio/mp3"
            )
        else:
            st.error(error_message)  # Display the error message
