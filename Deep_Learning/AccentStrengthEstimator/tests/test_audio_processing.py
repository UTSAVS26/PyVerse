"""
Tests for audio processing components.
"""

import pytest
import numpy as np
import os
import tempfile
from unittest.mock import Mock, patch

# Add src to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Try to import modules that require external dependencies
try:
    from src.audio.recorder import AudioRecorder
    from src.audio.processor import AudioProcessor
    HAS_AUDIO_DEPENDENCIES = True
except ImportError:
    HAS_AUDIO_DEPENDENCIES = False
    # Create mock classes for testing
    class AudioRecorder:
        def __init__(self, sample_rate=22050, channels=1):
            self.sample_rate = sample_rate
            self.channels = channels
            self.recording = False
            self.audio_data = []
        
        def get_audio_devices(self):
            return [{"name": "Mock Device", "max_inputs": 1}]
        
        def save_audio(self, audio_data, filename):
            return True
    
    class AudioProcessor:
        def __init__(self, sample_rate=22050):
            self.sample_rate = sample_rate
        
        def normalize_audio(self, audio):
            return audio
        
        def apply_preemphasis(self, audio, coef=0.97):
            return audio
        
        def segment_audio(self, audio, segment_length=0.025, hop_length=0.010):
            return [audio[:1000]]
        
        def compute_similarity(self, features1, features2):
            if len(features1) == 0 or len(features2) == 0:
                return 0.0
            return 0.8
        
        def load_audio(self, filename):
            return np.random.rand(1000), self.sample_rate

from src.audio.reference_generator import ReferenceGenerator


class TestAudioRecorder:
    """Test cases for AudioRecorder class."""
    
    @pytest.mark.skipif(not HAS_AUDIO_DEPENDENCIES, reason="Audio dependencies not available")
    def test_init(self):
        """Test AudioRecorder initialization."""
        recorder = AudioRecorder(sample_rate=44100, channels=2)
        assert recorder.sample_rate == 44100
        assert recorder.channels == 2
        assert recorder.recording == False
        assert recorder.audio_data == []
    
    @pytest.mark.skipif(not HAS_AUDIO_DEPENDENCIES, reason="Audio dependencies not available")
    def test_get_audio_devices(self):
        """Test getting audio devices."""
        recorder = AudioRecorder()
        devices = recorder.get_audio_devices()
        assert isinstance(devices, list)
        # Should have at least one input device
        assert len(devices) > 0
    
    @pytest.mark.skipif(not HAS_AUDIO_DEPENDENCIES, reason="Audio dependencies not available")
    def test_save_audio(self):
        """Test saving audio data."""
        recorder = AudioRecorder()
        audio_data = np.random.rand(1000, 1).astype(np.float32)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            filename = tmp_file.name
        
        try:
            success = recorder.save_audio(audio_data, filename)
            assert success == True
            assert os.path.exists(filename)
        finally:
            if os.path.exists(filename):
                os.unlink(filename)
    
    @pytest.mark.skipif(not HAS_AUDIO_DEPENDENCIES, reason="Audio dependencies not available")
    def test_save_audio_invalid_data(self):
        """Test saving invalid audio data."""
        recorder = AudioRecorder()
        success = recorder.save_audio(np.array([]), "test.wav")
        assert success == False


class TestAudioProcessor:
    """Test cases for AudioProcessor class."""
    
    @pytest.mark.skipif(not HAS_AUDIO_DEPENDENCIES, reason="Audio dependencies not available")
    def test_init(self):
        """Test AudioProcessor initialization."""
        processor = AudioProcessor(sample_rate=44100)
        assert processor.sample_rate == 44100
    
    @pytest.mark.skipif(not HAS_AUDIO_DEPENDENCIES, reason="Audio dependencies not available")
    def test_normalize_audio(self):
        """Test audio normalization."""
        processor = AudioProcessor()
        
        # Test with normal audio data
        audio = np.array([0.5, -0.3, 0.8, -0.9])
        normalized = processor.normalize_audio(audio)
        assert np.max(np.abs(normalized)) <= 1.0
        
        # Test with already normalized audio
        audio = np.array([0.1, -0.1, 0.05])
        normalized = processor.normalize_audio(audio)
        assert np.allclose(audio, normalized)
    
    @pytest.mark.skipif(not HAS_AUDIO_DEPENDENCIES, reason="Audio dependencies not available")
    def test_apply_preemphasis(self):
        """Test pre-emphasis filter."""
        processor = AudioProcessor()
        audio = np.array([1.0, 0.5, 0.25, 0.125])
        emphasized = processor.apply_preemphasis(audio, coef=0.97)
        
        assert len(emphasized) == len(audio)
        assert emphasized[0] == audio[0]  # First sample unchanged
    
    @pytest.mark.skipif(not HAS_AUDIO_DEPENDENCIES, reason="Audio dependencies not available")
    def test_segment_audio(self):
        """Test audio segmentation."""
        processor = AudioProcessor(sample_rate=22050)
        audio = np.random.rand(22050)  # 1 second of audio
        
        segments = processor.segment_audio(audio, segment_length=0.025, hop_length=0.010)
        
        assert len(segments) > 0
        segment_length_samples = int(0.025 * processor.sample_rate)
        assert len(segments[0]) == segment_length_samples
    
    @pytest.mark.skipif(not HAS_AUDIO_DEPENDENCIES, reason="Audio dependencies not available")
    def test_compute_similarity(self):
        """Test similarity computation."""
        processor = AudioProcessor()
        
        # Test identical features
        features1 = np.array([1, 2, 3, 4, 5])
        features2 = np.array([1, 2, 3, 4, 5])
        similarity = processor.compute_similarity(features1, features2)
        assert similarity == 1.0
        
        # Test different features
        features1 = np.array([1, 2, 3, 4, 5])
        features2 = np.array([5, 4, 3, 2, 1])
        similarity = processor.compute_similarity(features1, features2)
        assert 0.0 <= similarity <= 1.0
        
        # Test empty features
        similarity = processor.compute_similarity(np.array([]), np.array([1, 2, 3]))
        assert similarity == 0.0
    
    @pytest.mark.skipif(not HAS_AUDIO_DEPENDENCIES, reason="Audio dependencies not available")
    @patch('librosa.load')
    def test_load_audio(self, mock_load):
        """Test loading audio file."""
        processor = AudioProcessor()
        mock_load.return_value = (np.random.rand(1000), 22050)
        
        audio, sr = processor.load_audio("test.wav")
        assert len(audio) == 1000
        assert sr == 22050
    
    @pytest.mark.skipif(not HAS_AUDIO_DEPENDENCIES, reason="Audio dependencies not available")
    @patch('librosa.load')
    def test_load_audio_error(self, mock_load):
        """Test loading audio file with error."""
        processor = AudioProcessor()
        mock_load.side_effect = Exception("File not found")
        
        audio, sr = processor.load_audio("nonexistent.wav")
        assert len(audio) == 0
        assert sr == processor.sample_rate


class TestReferenceGenerator:
    """Test cases for ReferenceGenerator class."""
    
    def test_init(self):
        """Test ReferenceGenerator initialization."""
        generator = ReferenceGenerator(sample_rate=44100)
        assert generator.sample_rate == 44100
        assert generator.reference_data == {}
    
    def test_load_reference_phrases(self):
        """Test loading reference phrases."""
        generator = ReferenceGenerator()
        
        # Create temporary phrases file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
            tmp_file.write("Hello world.\nHow are you?\n")
            tmp_file.flush()
            
            phrases = generator.load_reference_phrases(tmp_file.name)
            assert len(phrases) == 2
            assert phrases[0] == "Hello world."
            assert phrases[1] == "How are you?"
        
        # Clean up
        os.unlink(tmp_file.name)
    
    def test_load_reference_phrases_nonexistent(self):
        """Test loading nonexistent phrases file."""
        generator = ReferenceGenerator()
        phrases = generator.load_reference_phrases("nonexistent.txt")
        assert phrases == []
    
    def test_generate_phonemes(self):
        """Test phoneme generation."""
        generator = ReferenceGenerator()
        
        # Test simple text
        phonemes = generator.generate_phonemes("hello")
        assert isinstance(phonemes, list)
        assert len(phonemes) > 0
        
        # Test text with special characters
        phonemes = generator.generate_phonemes("hello world!")
        assert isinstance(phonemes, list)
        assert len(phonemes) > 0
    
    def test_create_reference_data(self):
        """Test creating reference data."""
        generator = ReferenceGenerator()
        phrases = ["Hello world.", "How are you?"]
        
        reference_data = generator.create_reference_data(phrases)
        
        assert len(reference_data) == 2
        assert "phrase_1" in reference_data
        assert "phrase_2" in reference_data
        
        # Check structure
        phrase_data = reference_data["phrase_1"]
        assert "text" in phrase_data
        assert "phonemes" in phrase_data
        assert "phoneme_count" in phrase_data
        assert "word_count" in phrase_data
        assert "difficulty_level" in phrase_data
    
    def test_assess_difficulty(self):
        """Test difficulty assessment."""
        generator = ReferenceGenerator()
        
        # Test easy phrase
        easy_phrase = "Hi"
        easy_phonemes = ["h", "aɪ"]
        difficulty = generator._assess_difficulty(easy_phrase, easy_phonemes)
        assert difficulty == "easy"
        
        # Test hard phrase
        hard_phrase = "The quick brown fox jumps over the lazy dog"
        hard_phonemes = ["ð", "ə", "kwɪk", "braʊn", "fɑks", "dʒʌmps", "oʊvər", "ðə", "leɪzi", "dɔɡ"]
        difficulty = generator._assess_difficulty(hard_phrase, hard_phonemes)
        assert difficulty == "hard"
    
    def test_save_and_load_reference_data(self):
        """Test saving and loading reference data."""
        generator = ReferenceGenerator()
        phrases = ["Hello world.", "How are you?"]
        reference_data = generator.create_reference_data(phrases)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp_file:
            filename = tmp_file.name
        
        try:
            # Save
            success = generator.save_reference_data(filename)
            assert success == True
            
            # Load
            new_generator = ReferenceGenerator()
            success = new_generator.load_reference_data(filename)
            assert success == True
            
            # Compare
            assert new_generator.reference_data == reference_data
        finally:
            if os.path.exists(filename):
                os.unlink(filename)
    
    def test_get_phrase_data(self):
        """Test getting phrase data."""
        generator = ReferenceGenerator()
        phrases = ["Hello world."]
        generator.create_reference_data(phrases)
        
        phrase_data = generator.get_phrase_data("phrase_1")
        assert phrase_data["text"] == "Hello world."
        
        # Test nonexistent phrase
        phrase_data = generator.get_phrase_data("nonexistent")
        assert phrase_data == {}
    
    def test_get_phrases_by_difficulty(self):
        """Test getting phrases by difficulty."""
        generator = ReferenceGenerator()
        phrases = ["Hi", "Hello world.", "The quick brown fox jumps over the lazy dog"]
        generator.create_reference_data(phrases)
        
        easy_phrases = generator.get_phrases_by_difficulty("easy")
        assert len(easy_phrases) > 0
        
        hard_phrases = generator.get_phrases_by_difficulty("hard")
        assert len(hard_phrases) > 0


if __name__ == "__main__":
    pytest.main([__file__])
