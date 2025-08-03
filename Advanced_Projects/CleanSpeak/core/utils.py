import numpy as np

def rms_db(audio):
    rms = np.sqrt(np.mean(np.square(audio)))
    db = 20 * np.log10(rms + 1e-8)
    return db

def list_audio_devices():
    import sounddevice as sd
    return sd.query_devices() 