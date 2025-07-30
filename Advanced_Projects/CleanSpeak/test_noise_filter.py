import unittest
import numpy as np
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, '.')
from core.noise_filter import DeepFilterNetONNX, RNNoiseFilter, DemucsFilter

class TestNoiseFilters(unittest.TestCase):
    def setUp(self):
        self.dummy_audio = np.random.randn(1024).astype(np.float32)

    def test_deepfilternet(self):
        with patch.dict('sys.modules', {'onnxruntime': MagicMock()}):
            filt = DeepFilterNetONNX('dummy.onnx')
            filt.session.get_inputs.return_value = [type('I', (), {'name': 'input'})()]
            filt.session.get_outputs.return_value = [type('O', (), {'name': 'output'})()]
            filt.session.run.return_value = [self.dummy_audio[None, :]]
            out = filt.process(self.dummy_audio)
            self.assertEqual(out.shape, self.dummy_audio.shape)

    def test_rnnoise(self):
        with patch.dict('sys.modules', {'rnnoise': MagicMock()}):
            import rnnoise
            rnnoise.RNNoise.return_value.process.side_effect = lambda x: x
            filt = RNNoiseFilter()
            out = filt.process(self.dummy_audio)
            self.assertEqual(out.shape, self.dummy_audio.shape)

    def test_demucs(self):
        with patch.dict('sys.modules', {
            'torch': MagicMock(),
            'torchaudio': MagicMock(),
            'torchaudio.transforms': MagicMock(),
            'demucs': MagicMock(),
            'demucs.pretrained': MagicMock()
        }):
            import torch
            import demucs
            import demucs.pretrained
            import torchaudio
            import torchaudio.transforms
            # Mock Demucs model
            demucs.pretrained.get_model.return_value.eval.return_value = None
            demucs.pretrained.get_model.return_value.sources = ['vocals']
            demucs.pretrained.get_model.return_value.to.return_value = None
            torch.cuda.is_available.return_value = False
            dummy_out = np.random.randn(1, 1, 1024).astype(np.float32)
            demucs.pretrained.get_model.return_value.__call__ = lambda x: dummy_out
            # Patch Resample to return a mock with squeeze(0).numpy() chain
            class FakeNumpy:
                def numpy(self):
                    return np.random.randn(1024).astype(np.float32)
            class FakeTensor:
                def squeeze(self, axis):
                    return FakeNumpy()
            class FakeResample:
                def __call__(self, x):
                    return FakeTensor()
            torchaudio.transforms.Resample.return_value = FakeResample()
            filt = DemucsFilter()
            filt.model = demucs.pretrained.get_model.return_value
            out = filt.process(self.dummy_audio)
            self.assertIsInstance(out, np.ndarray)
            self.assertEqual(len(out), 1024)

if __name__ == '__main__':
    unittest.main() 