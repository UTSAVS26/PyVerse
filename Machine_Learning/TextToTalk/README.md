# TextToTalk: A PDF to MP3 Converter Web App

This is a Streamlit-based web application that allows users to upload a PDF file and convert its text content to an MP3 audio file. The application uses `pdfminer.six` for extracting text from the PDF and `gtts` (Google Text-to-Speech) for converting text to speech.

## Features

- Upload a PDF file using a drag-and-drop interface or a file upload button.
- Select the desired language for text-to-speech conversion.
- Convert the text content of the PDF to an MP3 audio file.
- Play the generated audio file directly on the web app.
- Download the generated audio file.

## Requirements

- `Python 3.6 or higher`
- `pdfminer.six`
- `gtts`
- `streamlit`

## Installation

#### First Fork the repository and then follow the steps given below!

1. Clone the repository in your local machine:
   ```sh
   git clone https://github.com/<your-username>/PyVerse.git
   cd Machine_Learning/TextToTalk
   ```

2. Create a virtual environment:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
   
3. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

  1. Run the Streamlit app:
     ```sh
     cd scripts
     streamlit run app.py
     ```

  2. Open your web browser and go to `http://localhost:8501` to access the app.

  3. Upload a PDF file using the provided upload button or drag and drop the file into the designated area.

  4. Select the desired language for the text-to-speech conversion.

  5. The app will extract the text from the uploaded PDF, convert it to speech, and display an audio player for you to listen to the generated MP3 file.

  6. You can also download the generated MP3 file using the download button.

## File Structure
- `ExtText.py`: Contains the function for extracting text from the uploaded PDF file using pdfminer.six.
- `TTS.py`: Contains the function for converting text to speech using gtts.
- `Pipeline.py`: Integrates the text extraction and text-to-speech conversion functions into a single pipeline.
- `app.py`: The main Streamlit app that provides the web interface for the PDF to MP3 conversion.

## Contributing
Contributions are welcome! If you find any bugs or have suggestions for improvements, please open an issue or create a pull request.

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.

