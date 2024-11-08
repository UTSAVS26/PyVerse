def pipeline(uploaded_file, lang='en'):
    """Extracts text and converts it to speech in a pipeline."""
    extracted_text, error_message = extract_text(uploaded_file)
    if error_message:
        return None, error_message

    try:
        return text_to_speech(extracted_text, lang=lang), None
    except Exception as e:
        return None, f"Error generating audio: {e}. Please try again."
