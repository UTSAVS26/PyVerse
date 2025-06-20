import queue
import sounddevice as sd
import numpy as np
import json
from faster_whisper import WhisperModel
from file_navigation import FileNavigator
import sys

navigator = FileNavigator()

model_size = "medium.en"
model = WhisperModel(model_size, device="cuda", compute_type="float16")

audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())


def transcribe_chunk(model, audio_chunk, sample_rate=16000):
    audio = audio_chunk.flatten().astype(np.float32) / 32768.0
    segments, _ = model.transcribe(audio, language="en", beam_size=1)
    text = ""
    for segment in segments:
        text += segment.text
    return text

with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                       channels=1, callback=audio_callback):
    print("🎤 Speak into the mic (Ctrl+C to stop):", flush=True)
    try:
        while True:
            print("Waiting for audio data...", flush=True)
            data = audio_queue.get()

            print("Audio data received.", flush=True)
            recognized_text = transcribe_chunk(model, data)

            print(f"Recognized: {recognized_text}", flush=True)
            result_json = json.dumps({"text": recognized_text})
            
            print(f"Passing to navigator: {result_json}", flush=True)
            navigator.parse_command(result_json)
    except KeyboardInterrupt:
        print("\n🛑 Stopped.", flush=True)
    except Exception as e:
        print(f"[ERROR] Exception occurred: {e}", file=sys.stderr, flush=True)