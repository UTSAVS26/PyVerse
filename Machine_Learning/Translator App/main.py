import streamlit as st
from translation import capture_and_translate
from utils import LANGUAGES
import time
import os

# Load custom CSS
def load_css():
    with open("assets/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# UI Structure
def main():
    st.title("🌐Real-Time Language Translator")
    st.markdown("Translate spoken language into other languages in real-time with a sleek experience.")

    load_css()  # Load custom styling

    # Language selection
    source_lang_name = st.selectbox("🌍 Select Source Language", list(LANGUAGES.keys()))
    target_lang_name = st.selectbox("🔄 Select Target Language", list(LANGUAGES.keys()))

    source_lang = LANGUAGES[source_lang_name]
    target_lang = LANGUAGES[target_lang_name]

    # Button to start listening
    if st.button("🎤 Start Listening", key="listen_button"):
        audio_file = capture_and_translate(source_lang, target_lang)
        if audio_file:
            time.sleep(1)  # Ensure pygame cleanup
            try:
                os.remove(audio_file)
            except Exception as e:
                st.error(f"⚠️ Error while deleting the file: {str(e)}")

if __name__ == "__main__":
    main()
