import unittest
from unittest.mock import patch, MagicMock

class TestAudioIO(unittest.TestCase):
    def test_audio_input_stream(self):
        with patch.dict('sys.modules', {'sounddevice': MagicMock()}):
            from core.audio_input import AudioInput
            import sounddevice
            ai = AudioInput(samplerate=16000, blocksize=256, device=1, channels=1)
            s = ai.stream()
            sounddevice.InputStream.assert_called_with(samplerate=16000, blocksize=256, device=1, channels=1, dtype='float32')

    def test_audio_output_stream(self):
        with patch.dict('sys.modules', {'sounddevice': MagicMock()}):
            from core.audio_output import AudioOutput
            import sounddevice
            ao = AudioOutput(samplerate=16000, blocksize=256, device=1, channels=1)
            s = ao.stream()
            sounddevice.OutputStream.assert_called_with(samplerate=16000, blocksize=256, device=1, channels=1, dtype='float32')

if __name__ == '__main__':
    unittest.main() 