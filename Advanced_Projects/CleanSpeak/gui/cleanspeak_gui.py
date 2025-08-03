import gradio as gr
import numpy as np
from core.noise_filter import get_filter
from core.utils import rms_db

# Dummy audio processing for GUI demo

def process_audio(audio, model_name, model_path):
    if audio is None:
        return None, "No input"
    # audio: (samples, 1)
    audio = np.array(audio[0])
    noise_filter = get_filter(model_name, model_path)
    filtered = noise_filter.process(audio)
    db = rms_db(filtered)
    return (filtered, 48000), f"Filtered dB: {db:.1f}"

def launch_gui():
    with gr.Blocks() as demo:
        gr.Markdown("# CleanSpeak: Real-Time AI Noise Filter")
        with gr.Row():
            model = gr.Dropdown(["deepfilternet", "rnnoise", "demucs"], value="deepfilternet", label="Model")
            model_path = gr.Textbox(value="models/deepfilternet.onnx", label="Model Path")
            virtualmic = gr.Checkbox(label="Output to Virtual Mic (stub)")
        audio_in = gr.Audio(source="microphone", type="numpy", label="Input Audio")
        audio_out = gr.Audio(label="Filtered Output")
        db_label = gr.Textbox(label="Noise Level (dB)")
        btn = gr.Button("Filter Noise")
        btn.click(process_audio, inputs=[audio_in, model, model_path], outputs=[audio_out, db_label])
    demo.launch()

if __name__ == "__main__":
    launch_gui() 