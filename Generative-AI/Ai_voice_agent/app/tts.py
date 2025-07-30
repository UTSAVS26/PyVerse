import os
import requests

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

def text_to_speech(text: str, output_path="output/output.mp3") -> str:
    """
    Convert text to speech using ElevenLabs API.
    Saves output as an mp3 file and returns the file path.
    """
    url = "https://api.elevenlabs.io/v1/text-to-speech"  # replace with actual ElevenLabs TTS endpoint
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }
    json_data = {
        "text": text,
        "voice": "some-voice-id",
        "format": "mp3"
    }
    response = requests.post(url, json=json_data, headers=headers)
    response.raise_for_status()

    with open(output_path, "wb") as f:
        f.write(response.content)

    return output_path
