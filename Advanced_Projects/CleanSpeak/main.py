import argparse
import os
import sys
import numpy as np
from core.audio_input import AudioInput
from core.noise_filter import get_filter
from core.audio_output import AudioOutput, VirtualMicOutput
from core.utils import rms_db, list_audio_devices

def main():
    parser = argparse.ArgumentParser(description="CleanSpeak: Real-Time AI Noise Filter")
    parser.add_argument('--model', choices=['rnnoise', 'deepfilternet', 'demucs'], default='deepfilternet')
    parser.add_argument('--model-path', type=str, default='models/deepfilternet.onnx')
    parser.add_argument('--virtualmic', choices=['enable', 'disable'], default='disable')
    parser.add_argument('--device', type=str, default=None, help='Input/output device name')
    parser.add_argument('--gui', action='store_true', help='Launch Gradio GUI')
    parser.add_argument('--pyqt-gui', action='store_true', help='Launch PyQt5 GUI')
    args = parser.parse_args()

    if args.gui:
        from gui.cleanspeak_gui import launch_gui
        launch_gui()
        return
    if args.pyqt_gui:
        from gui.cleanspeak_pyqt_gui import launch_pyqt_gui
        launch_pyqt_gui()
        return

    print(f"🎤 Capturing mic input...")
    print(f"🔇 Filtering background noise using {args.model}")
    if args.virtualmic == 'enable':
        print(f"🎧 Outputting to virtual mic: CleanSpeak_Virtual")
    else:
        print(f"🎧 Outputting to speakers/headphones")

    # Audio setup
    samplerate = 48000
    blocksize = 1024
    channels = 1

    # Device selection
    device = args.device
    if device:
        # Find device index by name
        devices = list_audio_devices()
        idx = next((i for i, d in enumerate(devices) if device.lower() in d['name'].lower()), None)
        if idx is None:
            print(f"Device '{device}' not found. Using default.")
            device = None
        else:
            device = idx

    # Model
    if args.model == "deepfilternet":
        if not os.path.exists(args.model_path):
            print(f"Model file not found: {args.model_path}")
            sys.exit(1)
        noise_filter = get_filter("deepfilternet", args.model_path)
    else:
        noise_filter = get_filter(args.model)

    # Audio streams
    input_stream = AudioInput(samplerate, blocksize, device, channels).stream()
    if args.virtualmic == 'enable':
        output_stream = VirtualMicOutput(samplerate, blocksize, device, channels).stream()
    else:
        output_stream = AudioOutput(samplerate, blocksize, device, channels).stream()

    try:
        with input_stream, output_stream:
            print("[Press Ctrl+C to stop]")
            while True:
                indata, _ = input_stream.read(blocksize)
                filtered = noise_filter.process(indata.flatten())
                output_stream.write(filtered.reshape(-1, channels))
                # Optional: print dB level
                db = rms_db(filtered)
                print(f"Noise-filtered level: {db:.1f} dB", end='\r')
    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 