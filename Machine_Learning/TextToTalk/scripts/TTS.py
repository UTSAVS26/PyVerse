from gtts import gTTS
from io import BytesIO

def text_to_speech(text, lang='en'):
    """Converts the extracted text to audio (MP3) using gTTS."""
    if not text:
        return None

    try:
        tts = gTTS(text=text, lang=lang)
        audio_file = BytesIO()
        tts.write_to_fp(audio_file)
        audio_file.seek(0)
        return audio_file
    except Exception as e:
        print(f"Error generating audio: {e}")
        return None
