import sys
import numpy as np
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel, QPushButton, QComboBox, QVBoxLayout, QWidget, QHBoxLayout
from PyQt5.QtCore import QThread, pyqtSignal
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from core.noise_filter import get_filter
from core.utils import list_audio_devices, rms_db

class AudioThread(QThread):
    update_waveform = pyqtSignal(np.ndarray)
    update_db = pyqtSignal(float)
    def __init__(self, model_name, model_path, device, parent=None):
        super().__init__(parent)
        self.model_name = model_name
        self.model_path = model_path
        self.device = device
        self.running = False
        self.samplerate = 48000
        self.blocksize = 1024
        self.channels = 1
    def run(self):
        self.running = True
        noise_filter = get_filter(self.model_name, self.model_path)
        input_stream = sd.InputStream(samplerate=self.samplerate, blocksize=self.blocksize, device=self.device, channels=self.channels, dtype='float32')
        output_stream = sd.OutputStream(samplerate=self.samplerate, blocksize=self.blocksize, device=self.device, channels=self.channels, dtype='float32')
        with input_stream, output_stream:
            while self.running:
                indata, _ = input_stream.read(self.blocksize)
                filtered = noise_filter.process(indata.flatten())
                output_stream.write(filtered.reshape(-1, self.channels))
                self.update_waveform.emit(filtered)
                self.update_db.emit(rms_db(filtered))
    def stop(self):
        self.running = False
        self.wait()

class WaveformCanvas(FigureCanvas):
    def __init__(self, parent=None):
        fig, self.ax = plt.subplots(figsize=(5,2), dpi=80)
        super().__init__(fig)
        self.setParent(parent)
        self.line, = self.ax.plot([], [])
        self.ax.set_ylim(-1, 1)
        self.ax.set_xlim(0, 1024)
        self.ax.set_title('Filtered Audio Waveform')
    def update_waveform(self, data):
        self.line.set_data(np.arange(len(data)), data)
        self.ax.set_xlim(0, len(data))
        self.draw()

class CleanSpeakGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('CleanSpeak - Real-Time AI Noise Filter')
        self.setGeometry(100, 100, 700, 400)
        self.model_label = QLabel('Model:')
        self.model_combo = QComboBox()
        self.model_combo.addItems(['deepfilternet', 'rnnoise', 'demucs'])
        self.device_label = QLabel('Device:')
        self.device_combo = QComboBox()
        self.populate_devices()
        self.start_btn = QPushButton('Start')
        self.stop_btn = QPushButton('Stop')
        self.stop_btn.setEnabled(False)
        self.db_label = QLabel('dB: ---')
        self.waveform = WaveformCanvas(self)
        layout = QVBoxLayout()
        h1 = QHBoxLayout()
        h1.addWidget(self.model_label)
        h1.addWidget(self.model_combo)
        h1.addWidget(self.device_label)
        h1.addWidget(self.device_combo)
        h1.addWidget(self.start_btn)
        h1.addWidget(self.stop_btn)
        layout.addLayout(h1)
        layout.addWidget(self.waveform)
        layout.addWidget(self.db_label)
        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)
        self.start_btn.clicked.connect(self.start_audio)
        self.stop_btn.clicked.connect(self.stop_audio)
        self.audio_thread = None
    def populate_devices(self):
        devices = list_audio_devices()
        self.device_combo.addItem('Default', None)
        for i, d in enumerate(devices):
            self.device_combo.addItem(f"{i}: {d['name']}", i)
    def start_audio(self):
        model = self.model_combo.currentText()
# at the top of the file
import os

    def start_audio(self):
        model = self.model_combo.currentText()
-        model_path = 'models/deepfilternet.onnx' if model == 'deepfilternet' else None
+        model_path = None
+        if model == 'deepfilternet':
+            model_path = 'models/deepfilternet.onnx'
+            if not os.path.exists(model_path):
+                QtWidgets.QMessageBox.warning(
+                    self,
+                    "Model Not Found",
+                    f"DeepFilterNet model not found at {model_path}. Please download it first."
+                )
+                return
        device = self.device_combo.currentData()
        # … rest of the method …
        device = self.device_combo.currentData()
        self.audio_thread = AudioThread(model, model_path, device)
        self.audio_thread.update_waveform.connect(self.waveform.update_waveform)
        self.audio_thread.update_db.connect(lambda db: self.db_label.setText(f'dB: {db:.1f}'))
        self.audio_thread.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
    def stop_audio(self):
        if self.audio_thread:
            self.audio_thread.stop()
            self.audio_thread = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

def launch_pyqt_gui():
    app = QApplication(sys.argv)
    gui = CleanSpeakGUI()
    gui.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    launch_pyqt_gui() 