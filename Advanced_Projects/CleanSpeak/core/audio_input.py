import sounddevice as sd
import numpy as np

class AudioInput:
    def __init__(self, samplerate=48000, blocksize=1024, device=None, channels=1):
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.device = device
        self.channels = channels

    def stream(self):
        return sd.InputStream(
            samplerate=self.samplerate,
            blocksize=self.blocksize,
            device=self.device,
            channels=self.channels,
            dtype='float32'
        ) 