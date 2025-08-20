"""
Audio recording functionality for capturing user speech.
"""

import sounddevice as sd
import numpy as np
import wave
import os
from typing import Optional, Tuple
import threading
import time


class AudioRecorder:
    """Handles audio recording from microphone input."""
    
    def __init__(self, sample_rate: int = 22050, channels: int = 1):
        """
        Initialize the audio recorder.
        
        Args:
            sample_rate: Sampling rate in Hz (default: 22050)
            channels: Number of audio channels (default: 1 for mono)
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.recording = False
        self.audio_data = []
        self.recording_thread = None
        
    def start_recording(self) -> None:
        """Start recording audio from microphone."""
        if self.recording:
            return
            
        self.recording = True
        self.audio_data = []
        
        def callback(indata, frames, time, status):
            if status:
                print(f"Status: {status}")
            if self.recording:
                self.audio_data.append(indata.copy())
        
        self.recording_thread = sd.InputStream(
            callback=callback,
            channels=self.channels,
            samplerate=self.sample_rate,
            dtype=np.float32
        )
        self.recording_thread.start()
        
    def stop_recording(self) -> np.ndarray:
        """
        Stop recording and return the recorded audio data.
        
        Returns:
            numpy.ndarray: Recorded audio data
        """
        if not self.recording:
            return np.array([])
            
        self.recording = False
        if self.recording_thread:
            self.recording_thread.stop()
            self.recording_thread.close()
            
        if self.audio_data:
            return np.concatenate(self.audio_data, axis=0)
        return np.array([])
    
    def record_for_duration(self, duration: float) -> np.ndarray:
        """
        Record audio for a specified duration.
        
        Args:
            duration: Recording duration in seconds
            
        Returns:
            numpy.ndarray: Recorded audio data
        """
        self.start_recording()
        time.sleep(duration)
        return self.stop_recording()
    
    def save_audio(self, audio_data: np.ndarray, filename: str) -> bool:
        """
        Save audio data to a WAV file.
        
        Args:
            audio_data: Audio data to save
            filename: Output filename
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            # Convert to 16-bit PCM
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            with wave.open(filename, 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_int16.tobytes())
            
            return True
        except Exception as e:
            print(f"Error saving audio: {e}")
            return False
    
    def get_audio_devices(self) -> list:
        """
        Get list of available audio input devices.
        
        Returns:
            list: List of available audio devices
        """
        devices = sd.query_devices()
        input_devices = []
        
        for i, device in enumerate(devices):
            if device['max_inputs'] > 0:
                input_devices.append({
                    'index': i,
                    'name': device['name'],
                    'channels': device['max_inputs']
                })
        
        return input_devices
    
    def set_input_device(self, device_index: int) -> bool:
        """
        Set the input device for recording.
        
        Args:
            device_index: Index of the input device
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            sd.default.device = device_index
            return True
        except Exception as e:
            print(f"Error setting input device: {e}")
            return False
