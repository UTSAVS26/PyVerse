from pathlib import Path
import queue
import sounddevice as sd
from vosk import Model, KaldiRecognizer
import pyaudio
from file_navigation import FileNavigator
import sys

# Load model using pathlib
model_path = Path.cwd() / "vosk-model-en-in-0.5"
assert model_path.exists(), f"Model not found at {model_path}"

model = Model(str(model_path))
recognizer = KaldiRecognizer(model, 16000)
audio_queue = queue.Queue()

# Callback to collect audio chunks
def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(bytes(indata))


navigator = FileNavigator()

# Open microphone stream
try:
    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                       channels=1, callback=audio_callback):
        print("🎤 Speak into the mic (Ctrl+C to stop):")
        try:
            while True:
                data = audio_queue.get()
                if recognizer.AcceptWaveform(data):
                    result = recognizer.Result()
                    # Parse the recognized text
                    print(result)
                    navigator.parse_command(result)
                
        except KeyboardInterrupt:
            print("\n🛑 Stopped.")
except Exception as e:
    print(f"[ERROR] Failed to initialize audio stream: {e}", file=sys.stderr, flush=True)
    sys.exit(1)
