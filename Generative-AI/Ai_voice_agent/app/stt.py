import os
from deepgram import Deepgram

DEEPGRAM_API_KEY = "62cea0425fd518402cb416d8716546d5acece0dc"
dg_client = Deepgram(DEEPGRAM_API_KEY)

async def transcribe(audio_url: str) -> str:
    """
    Transcribe audio from URL using Deepgram.
    """
    response = await dg_client.transcription.prerecorded(
        {"url": audio_url},
        {"punctuate": True, "language": "en"}
    )
    return response['results']['channels'][0]['alternatives'][0]['transcript']
