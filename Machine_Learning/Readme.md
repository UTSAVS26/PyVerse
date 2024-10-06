# 🌐 Real-Time Language Translator

A real-time language translation app built using Streamlit, Google Translate, and speech-to-text technology. This app allows users to speak in one language and get real-time translations in another, along with text-to-speech output for the translated text.

## Features
- **Speech Recognition:** Capture spoken input using a microphone.
- **Real-Time Translation:** Translate the captured speech into a chosen language.
- **Text-to-Speech:** Listen to the translated text in the target language.
- **Multiple Languages Supported:** Including English, Hindi, Tamil, Telugu, Marathi, Bengali, and more.

## Project Structure

streamlit_translation_app/ │ ├── main.py # Main Streamlit app file ├── translation.py # Core logic for speech capture, translation, and TTS ├── utils.py # Utility functions and language mappings ├── requirements.txt # Python dependencies ├── README.md # Project documentation └── assets/ └── styles.css # Custom CSS for UI


## Requirements
- Python 3.x
- A microphone for speech input

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/770navyasharma/Translator-app.git
    ```

2. Navigate to the project directory:

    ```bash
    cd Translator-app
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Run the app:

    ```bash
    streamlit run main.py
    ```

## How to Use

1. Select the source language (the language you will speak).
2. Select the target language (the language you want to translate to).
3. Click on **Start Listening** to capture speech.
4. Listen to the translated speech output and see the text on the screen.

## Customization

- **Language Support:** You can add more languages by updating the `LANGUAGES` dictionary in `utils.py`.
- **UI Styling:** Modify the `assets/styles.css` file to customize the look and feel of the app.

## Contributing

Feel free to contribute to this project by opening issues or submitting pull requests.

## License

This project is licensed under the MIT License. See `LICENSE` for more details.
