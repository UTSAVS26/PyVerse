from ExtText import extract_text
from TTS import text_to_speech

def pipeline(uploaded_file, lang='en'):
    """Extracts text and converts it to speech in a pipeline."""
    extracted_text = extract_text(uploaded_file)
    if extracted_text:
        return text_to_speech(extracted_text, lang=lang)
    else:
        return None
