from io import BytesIO
from pdfminer.high_level import extract_text_to_fp
from pdfminer.layout import LAParams

def extract_text(uploaded_file):
    """Extracts text from the uploaded PDF file using pdfminer.six."""
    if not uploaded_file:
        return None

    try:
        output_string = BytesIO()
        laparams = LAParams()
        extract_text_to_fp(uploaded_file, output_string, laparams=laparams)
        return output_string.getvalue().decode('utf-8')
    except Exception as e:
        print(f"Error extracting text: {e}")
        return None
