# Translator App

🎯 **Goal**
The goal of this project is to provide a real-time language translation system that captures speech, translates it into a chosen language, and plays the translated speech back to the user.

🧵 **Dataset**
No dataset is required for this project as it uses live speech capture and Google's translation services.

🧾 **Description**
This project allows users to translate spoken language into another language in real time. It captures the user’s voice input, translates it using Google’s translation API, and plays the translation back in audio format using Google Text-to-Speech (gTTS).

🧮 **What I had done!**
- Built a user-friendly interface with Streamlit for language selection and speech input.
- Used the `speech_recognition` library to capture live speech from the microphone.
- Implemented Google Translate for text translation between languages.
- Played back translated speech using `gTTS` and `pygame`.

🚀 **Models Implemented**
No machine learning models were used, but the project leverages Google Translate and Text-to-Speech APIs for translation and speech synthesis.

📚 **Libraries Needed**
- `Streamlit`
  
  - **Purpose**: Streamlit is an open-source framework to build web apps for data science and machine learning.
  - **Features**:
    - Turns Python scripts into interactive web applications.
    - Supports integration with multiple data visualization libraries (e.g., Matplotlib, Plotly).
    - Instantaneously updates the app during development with live reloading.
- `SpeechRecognition`

  - **Purpose**: A Python library for converting spoken language into written text.
  - **Features**:
    - Supports multiple speech recognition engines and APIs, including Google Web Speech API.
    - Can recognize audio from microphones, files, and other audio streams.
    - Handles speech in various formats such as `.wav`, `.aiff`, and more.

- `googletrans`

  - **Purpose**: Python library for interacting with the Google Translate API.
  - **Features**:
    - Provides translation between multiple languages.
    - Automatically detects the language of the input text.
    - Can retrieve translations and pronunciations of text in different languages.

- `gTTs`

  - **Purpose**: Python library for interfacing with Google’s Text-to-Speech API.
  - **Features**:
    - Converts text into spoken language with high-quality voice output.
    - Supports a wide range of languages and dialects.
    - Can save the output as `.mp3` files or play them directly in the application.
  
- `Pygame`

  - **Purpose**: A set of Python modules designed for multimedia application development, especially games.
  - **Features**:
    - Enables easy creation of 2D games with support for sound, graphics, and input.
    - Cross-platform compatibility (Windows, Mac, Linux).
    - Includes modules for handling images, sound, fonts, events, and input (keyboard, mouse, etc.).

📊 **Exploratory Data Analysis Results**
Not applicable for this project as it doesn’t involve data analysis.

📈 **Performance of the Models based on Accuracy Scores**
Not applicable.

📢 **Conclusion**
The Translator App offers a seamless, real-time translation experience. With a simple UI, users can quickly translate and hear the translated speech in a wide variety of languages.

✒️ **Your Signature**
Navya  
GitHub: https://github.com/770navyasharma
