"""
Audio recorder module for VoiceMoodMirror.
Handles microphone capture, buffering, and preprocessing.
"""

import pyaudio
import numpy as np
import threading
import time
import queue
from typing import Optional, Callable, List
import soundfile as sf


class AudioRecorder:
    """Real-time audio recorder with buffering and preprocessing capabilities."""
    
    def __init__(self, 
                 sample_rate: int = 22050,
                 chunk_size: int = 1024,
                 channels: int = 1,
                 format_type: int = pyaudio.paFloat32):
        """
        Initialize the audio recorder.
        
        Args:
            sample_rate: Audio sample rate in Hz
            chunk_size: Number of frames per buffer
            channels: Number of audio channels (1 for mono, 2 for stereo)
            format_type: Audio format type
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.format_type = format_type
        
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.recording_thread = None
        
        # Callback for when audio data is available
        self.on_audio_data: Optional[Callable] = None
        
    def start_recording(self, on_audio_data: Optional[Callable] = None):
        """
        Start recording audio from microphone.
        
        Args:
            on_audio_data: Optional callback function called with audio data
        """
        if self.is_recording:
            return
            
        self.on_audio_data = on_audio_data
        self.is_recording = True
        
        # Open audio stream
        self.stream = self.audio.open(
            format=self.format_type,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._audio_callback
        )
        
        self.stream.start_stream()
        
    def stop_recording(self):
        """Stop recording audio."""
        if not self.is_recording:
            return
            
        self.is_recording = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
            
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback function for audio stream."""
        if self.is_recording:
            # Convert bytes to numpy array
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            
            # Add to queue
            self.audio_queue.put(audio_data)
            
            # Call callback if provided
            if self.on_audio_data:
                self.on_audio_data(audio_data)
                
        return (in_data, pyaudio.paContinue)
    
    def get_audio_chunk(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """
        Get the next audio chunk from the queue.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Audio data as numpy array or None if timeout
        """
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def record_for_duration(self, duration: float) -> np.ndarray:
        """
        Record audio for a specific duration.
        
        Args:
            duration: Recording duration in seconds
            
        Returns:
            Concatenated audio data
        """
        chunks = []
        start_time = time.time()
        
        self.start_recording()
        
        while time.time() - start_time < duration:
            chunk = self.get_audio_chunk(timeout=0.1)
            if chunk is not None:
                chunks.append(chunk)
                
        self.stop_recording()
        
        if chunks:
            return np.concatenate(chunks)
        else:
            return np.array([])
    
    def save_audio(self, audio_data: np.ndarray, filename: str):
        """
        Save audio data to a file.
        
        Args:
            audio_data: Audio data as numpy array
            filename: Output filename
        """
        sf.write(filename, audio_data, self.sample_rate)
    
    def load_audio(self, filename: str) -> np.ndarray:
        """
        Load audio data from a file.
        
        Args:
            filename: Input filename
            
        Returns:
            Audio data as numpy array
        """
        audio_data, sample_rate = sf.read(filename)
        
        # Resample if necessary
        if sample_rate != self.sample_rate:
            # Simple resampling (for production, use librosa.resample)
            ratio = self.sample_rate / sample_rate
            new_length = int(len(audio_data) * ratio)
            audio_data = np.interp(
                np.linspace(0, len(audio_data), new_length),
                np.arange(len(audio_data)),
                audio_data
            )
            
        return audio_data
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.stop_recording()
        if hasattr(self, 'audio'):
            self.audio.terminate()


class AudioBuffer:
    """Circular buffer for storing audio data."""
    
    def __init__(self, max_duration: float, sample_rate: int = 22050):
        """
        Initialize audio buffer.
        
        Args:
            max_duration: Maximum buffer duration in seconds
            sample_rate: Audio sample rate in Hz
        """
        self.max_samples = int(max_duration * sample_rate)
        self.buffer = np.zeros(self.max_samples)
        self.sample_rate = sample_rate
        self.write_index = 0
        self.is_full = False
        
    def add_audio(self, audio_data: np.ndarray):
        """
        Add audio data to the buffer.
        
        Args:
            audio_data: Audio data as numpy array
        """
        samples_to_write = len(audio_data)
        
        if self.write_index + samples_to_write <= self.max_samples:
            # Simple case: no wrapping
            self.buffer[self.write_index:self.write_index + samples_to_write] = audio_data
        else:
            # Wrapping case
            samples_before_wrap = self.max_samples - self.write_index
            samples_after_wrap = samples_to_write - samples_before_wrap
            
            self.buffer[self.write_index:] = audio_data[:samples_before_wrap]
            self.buffer[:samples_after_wrap] = audio_data[samples_before_wrap:]
            
        self.write_index = (self.write_index + samples_to_write) % self.max_samples
        
        if self.write_index == 0:
            self.is_full = True
            
    def get_latest_audio(self, duration: float) -> np.ndarray:
        """
        Get the latest audio data from the buffer.
        
        Args:
            duration: Duration in seconds
            
        Returns:
            Audio data as numpy array
        """
        samples = int(duration * self.sample_rate)
        samples = min(samples, self.max_samples)
        
        if self.is_full:
            # Buffer is full, get from write_index backwards
            start_idx = (self.write_index - samples) % self.max_samples
            if start_idx < self.write_index:
                return self.buffer[start_idx:self.write_index]
            else:
                return np.concatenate([
                    self.buffer[start_idx:],
                    self.buffer[:self.write_index]
                ])
        else:
            # Buffer not full, get from beginning to write_index
            return self.buffer[:self.write_index]
    
    def clear(self):
        """Clear the buffer."""
        self.buffer.fill(0)
        self.write_index = 0
        self.is_full = False
