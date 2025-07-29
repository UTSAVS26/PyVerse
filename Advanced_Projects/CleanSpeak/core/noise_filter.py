import numpy as np

class BaseNoiseFilter:
    def process(self, audio_block):
        raise NotImplementedError

class DeepFilterNetONNX(BaseNoiseFilter):
    def __init__(self, model_path):
        import onnxruntime as ort
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def process(self, audio_block):
        input_audio = audio_block.astype(np.float32)[None, :]
        output = self.session.run([self.output_name], {self.input_name: input_audio})[0]
        return output.flatten()

class RNNoiseFilter(BaseNoiseFilter):
    def __init__(self):
        import rnnoise
        self.rnnoise = rnnoise.RNNoise()

    def process(self, audio_block):
        import rnnoise
        frame_size = 480
        audio_block = audio_block.astype(np.float32)
        out = np.zeros_like(audio_block)
        for i in range(0, len(audio_block), frame_size):
            frame = audio_block[i:i+frame_size]
            if len(frame) < frame_size:
                frame = np.pad(frame, (0, frame_size - len(frame)))
            processed = self.rnnoise.process(frame)
            valid_len = min(len(frame), len(audio_block) - i)
            out[i:i+valid_len] = processed[:valid_len]
        return out[:len(audio_block)]

class DemucsFilter(BaseNoiseFilter):
    def __init__(self):
        import torch
        from demucs.pretrained import get_model as get_demucs_model
        self.model = get_demucs_model('htdemucs')
        self.model.eval()
        self.samplerate = 44100  # Demucs default
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

    def process(self, audio_block):
        import torch
        from torchaudio.transforms import Resample
        orig_sr = 48000
        if orig_sr != self.samplerate:
            resampler = Resample(orig_sr, self.samplerate)
            audio_block = resampler(torch.tensor(audio_block).unsqueeze(0)).squeeze(0).numpy()
        wav = torch.tensor(audio_block, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            sources = self.model(wav)
        if hasattr(self.model, 'sources') and 'vocals' in self.model.sources:
            idx = self.model.sources.index('vocals')
            clean = sources[:, idx, :].cpu().numpy().flatten()
        else:
            clean = sources[:, 0, :].cpu().numpy().flatten()
        if self.samplerate != orig_sr:
            resampler = Resample(self.samplerate, orig_sr)
            clean = resampler(torch.tensor(clean).unsqueeze(0)).squeeze(0).numpy()
        return clean[:len(audio_block)]

def get_filter(model_name, model_path=None):
    if model_name == "deepfilternet":
        return DeepFilterNetONNX(model_path)
    elif model_name == "rnnoise":
        return RNNoiseFilter()
    elif model_name == "demucs":
        return DemucsFilter()
    else:
        raise ValueError(f"Unknown model: {model_name}") 