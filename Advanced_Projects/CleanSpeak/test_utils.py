import unittest
import numpy as np
from unittest.mock import patch, MagicMock
from core.utils import rms_db, list_audio_devices

class TestUtils(unittest.TestCase):
    def test_rms_db(self):
        arr = np.ones(1000, dtype=np.float32) * 0.5
        db = rms_db(arr)
        self.assertTrue(isinstance(db, (float, np.floating)))
        self.assertTrue(np.isfinite(db))

    def test_list_audio_devices(self):
        with patch.dict('sys.modules', {'sounddevice': MagicMock()}):
            import sounddevice
            sounddevice.query_devices.return_value = [{'name': 'Fake Mic'}]
            devices = list_audio_devices()
            self.assertEqual(devices[0]['name'], 'Fake Mic')

if __name__ == '__main__':
    unittest.main() 