def extract_text(uploaded_file):
    """Extracts text from the uploaded PDF file using pdfminer.six."""
    if not uploaded_file:
        return None, "No file uploaded."

    try:
        output_string = BytesIO()
        laparams = LAParams()
        extract_text_to_fp(uploaded_file, output_string, laparams=laparams)
        return output_string.getvalue().decode('utf-8'), None
    except Exception as e:
        error_message = f"Error extracting text: {e}. Please ensure the PDF is not corrupted and is in a supported format."
        return None, error_message
