# CleanSpeak: Real-Time AI Noise Filter for Microphone Input

> Crystal-clear voice in any environment — powered by AI.

---

## Project Overview

**CleanSpeak** is a Python-based, cross-platform, real-time microphone noise suppression tool. It leverages state-of-the-art AI models (RNNoise, DeepFilterNet, Demucs) to filter out background noise from your microphone input, delivering clean audio for calls, streaming, and recordings. CleanSpeak can output the filtered audio to your speakers/headphones or to a **virtual microphone** for use in any application (OBS, Discord, Zoom, etc.).

---

## Key Features

- **Real-Time Audio Capture & Filtering:**  
  Captures microphone audio in real time and applies machine learning noise reduction with minimal latency.

- **AI-Powered Noise Suppression:**  
  Supports multiple models:
  - [RNNoise (Xiph)](https://github.com/xiph/rnnoise): Fast, C-based, low-latency.
  - [DeepFilterNet v2](https://github.com/Rikorose/DeepFilterNet): ONNX, neural pipeline, high quality.
  - [Demucs](https://github.com/facebookresearch/demucs): Deep speech separation, best for powerful machines.

- **Flexible Interfaces:**  
  - **Command-Line Interface (CLI)**
  - **PyQt5 GUI:** Modern desktop app with waveform and dB meter.
  - **Gradio GUI:** Web-based, easy for quick demos.

- **Virtual Microphone Output:**  
  Route clean audio to a virtual mic (VB-Cable, BlackHole, PulseAudio) for use in any app.

- **Noise Level Visualization:**  
  Real-time waveform and decibel meter.

- **Cross-Platform:**  
  Works on Windows, Linux, and macOS.

---

## Architecture & Tech Stack

| Component            | Technology                                     |
|----------------------|------------------------------------------------|
| Audio I/O            | `sounddevice`, `pyaudio`, `torchaudio`         |
| AI Models            | RNNoise, DeepFilterNet, Demucs, ONNX runtime   |
| GUI                  | `PyQt5`, `Gradio`                              |
| Virtual Mic Support  | VB-Cable (Windows), PulseAudio (Linux), BlackHole (macOS) |
| Backend              | Python 3.8+                                    |
| Testing              | `unittest`, `mock`                             |

---

## Folder Structure

```
cleanspeak/
├── models/                  # Place model files here (e.g., deepfilternet.onnx)
├── core/
│   ├── audio_input.py       # Microphone capture
│   ├── noise_filter.py      # ML inference (RNNoise, DeepFilterNet, Demucs)
│   ├── audio_output.py      # Playback / virtual mic
│   └── utils.py             # Utilities (RMS, device listing)
├── gui/
│   ├── cleanspeak_gui.py        # Gradio GUI
│   └── cleanspeak_pyqt_gui.py   # PyQt5 GUI
├── main.py                  # CLI entry point
├── requirements.txt
├── test_noise_filter.py     # Unit tests for noise filters
├── test_utils.py            # Unit tests for utilities
├── test_audio_io.py         # Unit tests for audio I/O
└── README.md
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/cleanspeak
cd cleanspeak
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. (Optional) Install Virtual Mic Driver

- **Windows:** [VB-Cable](https://vb-audio.com/Cable/)
- **Linux:** `pactl load-module module-null-sink`
- **macOS:** [BlackHole](https://github.com/ExistentialAudio/BlackHole)

### 4. Download/Place Model Files

- **DeepFilterNet:** Download the ONNX model and place it in `models/` (e.g., `models/deepfilternet.onnx`).
- **RNNoise:** Uses built-in weights via `python-rnnoise`.
- **Demucs:** Downloads weights automatically via `demucs` package.

---

## Usage

### **Command-Line Interface**

```bash
python main.py --model deepfilternet --virtualmic enable
```

**Options:**
- `--model [rnnoise|deepfilternet|demucs]`   AI model to use
- `--model-path PATH`                        Path to ONNX model (for DeepFilterNet)
- `--virtualmic [enable|disable]`            Output to virtual mic
- `--device DEVICE_NAME`                     Select specific audio device
- `--gui`                                    Launch Gradio GUI
- `--pyqt-gui`                               Launch PyQt5 GUI

### **PyQt5 GUI**

```bash
python main.py --pyqt-gui
```
- Select model, device, and start/stop filtering.
- View real-time waveform and dB meter.

### **Gradio GUI**

```bash
python main.py --gui
```
- Web-based interface for quick demos and testing.

---

## Model Details

### **RNNoise**
- C-based, fast, and effective for real-time use.
- Great for low-end hardware.
- No extra model download needed (uses `python-rnnoise`).

### **DeepFilterNet v2**
- Neural filter pipeline, ONNX format.
- Low latency, good quality.
- Download ONNX model from [DeepFilterNet releases](https://github.com/Rikorose/DeepFilterNet/releases) and place in `models/`.

### **Demucs**
- Deep learning speech separation (PyTorch).
- High quality, but higher latency and resource usage.
- Best for powerful machines.
- Model weights are downloaded automatically by the `demucs` package.

---

## Audio Device Selection

- Use `--device` to select a specific input/output device by name or index.
- Use the PyQt5 GUI for a dropdown device selector.
- List all devices in Python:

```python
import sounddevice as sd
print(sd.query_devices())
```

---

## Virtual Microphone Output

- To use CleanSpeak as a microphone in other apps (OBS, Discord, Zoom):
  1. Install a virtual audio cable/driver for your OS.
  2. Set CleanSpeak's output device to the virtual cable.
  3. Select the virtual cable as your mic in your target app.

---

## Visualization

- **PyQt5 GUI:** Real-time waveform and dB meter.
- **Gradio GUI:** Waveform and dB display after filtering.

---

## Testing & Development

- All core modules are covered by unit tests:
  - `test_noise_filter.py`: Tests for all noise filter classes (mocked models).
  - `test_utils.py`: Tests for utility functions.
  - `test_audio_io.py`: Tests for audio input/output classes.
- Run all tests:

```bash
python -m unittest discover
```

- **Note:** The Demucs test uses advanced mocking and may not fully simulate the real model chain due to the complexity of PyTorch/torchaudio APIs.

---

## Extending CleanSpeak

- Add new models by extending `core/noise_filter.py`.
- Add new GUIs or integrations in the `gui/` folder.
- Contribute tests for new features in the test scripts.

---