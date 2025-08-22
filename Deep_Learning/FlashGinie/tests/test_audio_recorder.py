"""
Tests for the audio recorder module.
"""

import pytest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio.recorder import AudioRecorder, AudioBuffer


class TestAudioRecorder:
    """Test cases for AudioRecorder class."""
    
    def setup_method(self):
        """Setup method called before each test."""
        # Mock PyAudio to avoid actual audio hardware dependencies
        self.mock_pyaudio = Mock()
        self.mock_stream = Mock()
        
        with patch('audio.recorder.pyaudio.PyAudio', return_value=self.mock_pyaudio):
            self.mock_pyaudio.open.return_value = self.mock_stream
            self.recorder = AudioRecorder()
    
    def test_initialization(self):
        """Test AudioRecorder initialization."""
        assert self.recorder.sample_rate == 22050
        assert self.recorder.chunk_size == 1024
        assert self.recorder.channels == 1
        assert self.recorder.is_recording == False
        assert self.recorder.stream is None
    
    def test_start_recording(self):
        """Test starting recording."""
        # Mock callback function
        mock_callback = Mock()
        
        self.recorder.start_recording(mock_callback)
        
        assert self.recorder.is_recording == True
        assert self.recorder.on_audio_data == mock_callback
        self.mock_pyaudio.open.assert_called_once()
        self.mock_stream.start_stream.assert_called_once()
    
    def test_start_recording_already_recording(self):
        """Test starting recording when already recording."""
        self.recorder.is_recording = True
        
        self.recorder.start_recording()
        
        # Should not call PyAudio.open again
        self.mock_pyaudio.open.assert_not_called()
    
    def test_stop_recording(self):
        """Test stopping recording."""
        self.recorder.is_recording = True
        self.recorder.stream = self.mock_stream
        
        self.recorder.stop_recording()
        
        assert self.recorder.is_recording == False
        self.mock_stream.stop_stream.assert_called_once()
        self.mock_stream.close.assert_called_once()
    
    def test_stop_recording_not_recording(self):
        """Test stopping recording when not recording."""
        self.recorder.is_recording = False
        
        self.recorder.stop_recording()
        
        # Should not call stream methods
        self.mock_stream.stop_stream.assert_not_called()
        self.mock_stream.close.assert_not_called()
    
    def test_audio_callback(self):
        """Test audio callback function."""
        mock_callback = Mock()
        self.recorder.is_recording = True
        self.recorder.on_audio_data = mock_callback
        
        # Create mock audio data
        mock_in_data = np.array([0.1, 0.2, 0.3], dtype=np.float32).tobytes()
        
        result = self.recorder._audio_callback(mock_in_data, 1024, {}, None)
        
        # Check that data was added to queue
        assert not self.recorder.audio_queue.empty()
        
        # Check that callback was called
        mock_callback.assert_called_once()
        
        # Check return value
        assert result == (mock_in_data, 1)  # paContinue
    
    def test_get_audio_chunk(self):
        """Test getting audio chunk from queue."""
        # Add test data to queue
        test_data = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        self.recorder.audio_queue.put(test_data)
        
        result = self.recorder.get_audio_chunk()
        
        assert np.array_equal(result, test_data)
    
    def test_get_audio_chunk_timeout(self):
        """Test getting audio chunk with timeout."""
        result = self.recorder.get_audio_chunk(timeout=0.1)
        
        assert result is None
    
    def test_record_for_duration(self):
        """Test recording for a specific duration."""
        # Mock time to control the loop
        with patch('time.time') as mock_time:
            mock_time.side_effect = [0.0, 0.5, 1.0, 1.5, 2.0]
            
            # Mock get_audio_chunk to return test data
            test_chunk = np.array([0.1, 0.2, 0.3], dtype=np.float32)
            self.recorder.get_audio_chunk = Mock(side_effect=[test_chunk, test_chunk, None])
            
            with patch.object(self.recorder, 'start_recording'), \
                 patch.object(self.recorder, 'stop_recording'):
                
                result = self.recorder.record_for_duration(1.5)
                
                # Should concatenate the chunks
                expected = np.concatenate([test_chunk, test_chunk])
                assert np.array_equal(result, expected)
    
    def test_save_and_load_audio(self):
        """Test saving and loading audio data."""
        # Create test audio data
        test_audio = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_filename = temp_file.name
        
        try:
            # Test saving
            self.recorder.save_audio(test_audio, temp_filename)
            
            # Test loading
            loaded_audio = self.recorder.load_audio(temp_filename)
            
            # Check that data is approximately equal (due to audio format conversion)
            assert len(loaded_audio) == len(test_audio)
            assert np.allclose(loaded_audio, test_audio, atol=0.1)
        
        finally:
            # Clean up
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)
    
    def test_cleanup(self):
        """Test cleanup when object is destroyed."""
        self.recorder.is_recording = True
        self.recorder.stream = self.mock_stream
        
        # Simulate object destruction
        self.recorder.__del__()
        
        # Should stop recording
        assert self.recorder.is_recording == False
        self.mock_stream.stop_stream.assert_called_once()
        self.mock_stream.close.assert_called_once()
        self.mock_pyaudio.terminate.assert_called_once()


class TestAudioBuffer:
    """Test cases for AudioBuffer class."""
    
    def setup_method(self):
        """Setup method called before each test."""
        self.buffer = AudioBuffer(max_duration=2.0, sample_rate=22050)
    
    def test_initialization(self):
        """Test AudioBuffer initialization."""
        assert self.buffer.max_samples == 44100  # 2.0 * 22050
        assert self.buffer.sample_rate == 22050
        assert self.buffer.write_index == 0
        assert self.buffer.is_full == False
        assert len(self.buffer.buffer) == 44100
    
    def test_add_audio_simple(self):
        """Test adding audio data without wrapping."""
        test_data = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        
        self.buffer.add_audio(test_data)
        
        assert self.buffer.write_index == 3
        assert np.array_equal(self.buffer.buffer[:3], test_data)
        assert self.buffer.is_full == False
    
    def test_add_audio_wrapping(self):
        """Test adding audio data with buffer wrapping."""
        # Fill buffer almost completely
        large_data = np.ones(44000, dtype=np.float32)
        self.buffer.add_audio(large_data)
        
        # Add more data to trigger wrapping
        wrap_data = np.array([0.5, 0.6, 0.7], dtype=np.float32)
        self.buffer.add_audio(wrap_data)
        
        # Check that wrapping occurred
        assert self.buffer.write_index == 3
        assert self.buffer.buffer[0] == 0.5
        assert self.buffer.buffer[1] == 0.6
        assert self.buffer.buffer[2] == 0.7
    
    def test_get_latest_audio_not_full(self):
        """Test getting latest audio when buffer is not full."""
        test_data = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
        self.buffer.add_audio(test_data)
        
        result = self.buffer.get_latest_audio(1.0)  # 1 second
        
        # Should return all available data
        assert np.array_equal(result, test_data)
    
    def test_get_latest_audio_full(self):
        """Test getting latest audio when buffer is full."""
        # Fill the buffer
        large_data = np.ones(44100, dtype=np.float32)
        self.buffer.add_audio(large_data)
        
        # Add more data to make it full
        extra_data = np.array([0.5, 0.6, 0.7], dtype=np.float32)
        self.buffer.add_audio(extra_data)
        
        result = self.buffer.get_latest_audio(1.0)  # 1 second
        
        # Should return the latest 22050 samples
        expected_samples = 22050
        assert len(result) == expected_samples
    
    def test_get_latest_audio_wrapping(self):
        """Test getting latest audio with wrapping."""
        # Fill buffer and add data to trigger wrapping
        large_data = np.ones(44000, dtype=np.float32)
        self.buffer.add_audio(large_data)
        
        wrap_data = np.array([0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float32)
        self.buffer.add_audio(wrap_data)
        
        result = self.buffer.get_latest_audio(0.0001)  # Very short duration
        
        # Should return the wrapped data
        assert len(result) > 0
        assert result[0] == 0.5
    
    def test_clear(self):
        """Test clearing the buffer."""
        # Add some data
        test_data = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        self.buffer.add_audio(test_data)
        
        # Clear the buffer
        self.buffer.clear()
        
        assert self.buffer.write_index == 0
        assert self.buffer.is_full == False
        assert np.all(self.buffer.buffer == 0)
    
    def test_buffer_full_flag(self):
        """Test that is_full flag is set correctly."""
        # Fill the buffer completely
        large_data = np.ones(44100, dtype=np.float32)
        self.buffer.add_audio(large_data)
        
        assert self.buffer.is_full == True
        assert self.buffer.write_index == 0  # Should wrap around
    
    def test_edge_cases(self):
        """Test edge cases."""
        # Test with empty data
        self.buffer.add_audio(np.array([], dtype=np.float32))
        assert self.buffer.write_index == 0
        
        # Test getting audio from empty buffer
        result = self.buffer.get_latest_audio(1.0)
        assert len(result) == 0
        
        # Test with duration longer than buffer
        self.buffer.add_audio(np.array([0.1, 0.2, 0.3], dtype=np.float32))
        result = self.buffer.get_latest_audio(10.0)  # 10 seconds
        assert len(result) == 3  # Should return all available data
