"""
Tests for the feature extractor module.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio.feature_extractor import FeatureExtractor


class TestFeatureExtractor:
    """Test cases for FeatureExtractor class."""
    
    def setup_method(self):
        """Setup method called before each test."""
        self.extractor = FeatureExtractor(sample_rate=22050)
        
        # Create mock audio data
        self.test_audio = np.random.randn(22050)  # 1 second of random audio
    
    def test_initialization(self):
        """Test FeatureExtractor initialization."""
        assert self.extractor.sample_rate == 22050
    
    def test_extract_features(self):
        """Test extracting all features from audio."""
        features = self.extractor.extract_features(self.test_audio)
        
        # Check that features dictionary is returned
        assert isinstance(features, dict)
        assert len(features) > 0
        
        # Check that all expected feature categories are present
        expected_categories = [
            'duration', 'rms_energy', 'zero_crossing_rate',
            'pitch_mean', 'pitch_std', 'energy_mean', 'energy_std',
            'spectral_centroid_mean', 'tempo', 'speech_rate'
        ]
        
        for category in expected_categories:
            assert category in features
            assert isinstance(features[category], (int, float))
    
    def test_extract_basic_features(self):
        """Test extracting basic audio features."""
        features = self.extractor._extract_basic_features(self.test_audio)
        
        assert 'duration' in features
        assert 'rms_energy' in features
        assert 'zero_crossing_rate' in features
        
        # Check duration calculation
        expected_duration = len(self.test_audio) / self.extractor.sample_rate
        assert abs(features['duration'] - expected_duration) < 0.001
        
        # Check RMS energy
        expected_rms = np.sqrt(np.mean(self.test_audio**2))
        assert abs(features['rms_energy'] - expected_rms) < 0.001
    
    def test_extract_pitch_features(self):
        """Test extracting pitch features."""
        # Mock librosa.piptrack to return predictable values
        mock_pitches = np.array([[100, 150, 200], [120, 180, 220]])
        mock_magnitudes = np.array([[0.5, 0.8, 0.3], [0.6, 0.9, 0.4]])
        
        with patch('librosa.piptrack', return_value=(mock_pitches, mock_magnitudes)):
            features = self.extractor._extract_pitch_features(self.test_audio)
        
        # Check that pitch features are present
        pitch_features = ['pitch_mean', 'pitch_std', 'pitch_min', 'pitch_max', 'pitch_range']
        for feature in pitch_features:
            assert feature in features
            assert isinstance(features[feature], (int, float))
    
    def test_extract_pitch_features_no_voiced(self):
        """Test pitch extraction with no voiced frames."""
        # Mock librosa.piptrack to return no voiced frames
        mock_pitches = np.array([[100, 150, 200], [120, 180, 220]])
        mock_magnitudes = np.array([[0.01, 0.02, 0.01], [0.01, 0.02, 0.01]])  # Very low magnitudes
        
        with patch('librosa.piptrack', return_value=(mock_pitches, mock_magnitudes)):
            features = self.extractor._extract_pitch_features(self.test_audio)
        
        # Should return zero values for pitch features
        pitch_features = ['pitch_mean', 'pitch_std', 'pitch_min', 'pitch_max', 'pitch_range']
        for feature in pitch_features:
            assert features[feature] == 0
    
    def test_extract_energy_features(self):
        """Test extracting energy features."""
        # Mock librosa.feature.rms to return predictable values
        mock_rms = np.array([0.1, 0.2, 0.15, 0.25, 0.18])
        
        with patch('librosa.feature.rms', return_value=[mock_rms]):
            features = self.extractor._extract_energy_features(self.test_audio)
        
        # Check that energy features are present
        energy_features = ['energy_mean', 'energy_std', 'energy_max', 'energy_min', 'energy_range']
        for feature in energy_features:
            assert feature in features
            assert isinstance(features[feature], (int, float))
        
        # Check calculations
        assert abs(features['energy_mean'] - np.mean(mock_rms)) < 0.001
        assert abs(features['energy_max'] - np.max(mock_rms)) < 0.001
        assert abs(features['energy_min'] - np.min(mock_rms)) < 0.001
    
    def test_extract_spectral_features(self):
        """Test extracting spectral features."""
        # Mock librosa spectral features
        mock_centroids = np.array([2000, 2200, 1800, 2400])
        mock_bandwidth = np.array([1000, 1200, 800, 1400])
        mock_rolloff = np.array([3000, 3200, 2800, 3400])
        mock_mfccs = np.random.randn(13, 10)  # 13 MFCCs, 10 frames
        
        with patch('librosa.feature.spectral_centroid', return_value=[mock_centroids]), \
             patch('librosa.feature.spectral_bandwidth', return_value=[mock_bandwidth]), \
             patch('librosa.feature.spectral_rolloff', return_value=[mock_rolloff]), \
             patch('librosa.feature.mfcc', return_value=mock_mfccs):
            
            features = self.extractor._extract_spectral_features(self.test_audio)
        
        # Check spectral features
        spectral_features = ['spectral_centroid_mean', 'spectral_centroid_std',
                           'spectral_bandwidth_mean', 'spectral_bandwidth_std',
                           'spectral_rolloff_mean', 'spectral_rolloff_std']
        
        for feature in spectral_features:
            assert feature in features
            assert isinstance(features[feature], (int, float))
        
        # Check MFCC features
        for i in range(1, 6):
            assert f'mfcc_{i}_mean' in features
            assert f'mfcc_{i}_std' in features
    
    def test_extract_tempo_features(self):
        """Test extracting tempo features."""
        # Mock librosa tempo and onset features
        mock_tempo = 120.0
        mock_onset_env = np.array([0.1, 0.3, 0.2, 0.4, 0.1])
        
        with patch('librosa.beat.beat_track', return_value=(mock_tempo, None)), \
             patch('librosa.onset.onset_strength', return_value=mock_onset_env):
            
            features = self.extractor._extract_tempo_features(self.test_audio)
        
        # Check tempo features
        assert 'tempo' in features
        assert 'onset_strength_mean' in features
        assert 'onset_strength_std' in features
        assert 'speech_rate' in features
        
        assert features['tempo'] == mock_tempo
        assert abs(features['onset_strength_mean'] - np.mean(mock_onset_env)) < 0.001
    
    def test_extract_voice_quality_features(self):
        """Test extracting voice quality features."""
        # Mock librosa spectral centroid
        mock_centroids = np.array([2000, 2200, 1800, 2400])
        
        with patch('librosa.feature.spectral_centroid', return_value=[mock_centroids]):
            features = self.extractor._extract_voice_quality_features(self.test_audio)
        
        # Check voice quality features
        assert 'voice_quality' in features
        assert 'jitter' in features
        assert 'shimmer' in features
        
        # Voice quality should be normalized
        assert 0 <= features['voice_quality'] <= 1
    
    def test_calculate_pitch_slope(self):
        """Test pitch slope calculation."""
        # Test with valid pitch data
        pitches = np.array([100, 120, 140, 160, 180])
        slope = self.extractor._calculate_pitch_slope(pitches)
        
        # Should be positive (increasing pitch)
        assert slope > 0
        
        # Test with single pitch
        single_pitch = np.array([150])
        slope = self.extractor._calculate_pitch_slope(single_pitch)
        assert slope == 0.0
    
    def test_calculate_energy_slope(self):
        """Test energy slope calculation."""
        # Test with valid energy data
        energy = np.array([0.1, 0.2, 0.15, 0.25, 0.3])
        slope = self.extractor._calculate_energy_slope(energy)
        
        # Should be positive (increasing energy)
        assert slope > 0
        
        # Test with single energy value
        single_energy = np.array([0.2])
        slope = self.extractor._calculate_energy_slope(single_energy)
        assert slope == 0.0
    
    def test_extract_features_windowed(self):
        """Test extracting features from windowed audio."""
        # Create longer audio for windowing
        long_audio = np.random.randn(22050 * 5)  # 5 seconds
        
        features_list = self.extractor.extract_features_windowed(long_audio, window_duration=2.0)
        
        # Should return multiple feature dictionaries
        assert isinstance(features_list, list)
        assert len(features_list) > 0
        
        # Each window should have the same features
        for features in features_list:
            assert isinstance(features, dict)
            assert len(features) > 0
    
    def test_get_feature_names(self):
        """Test getting feature names."""
        feature_names = self.extractor.get_feature_names()
        
        assert isinstance(feature_names, list)
        assert len(feature_names) > 0
        
        # Check that all names are strings
        for name in feature_names:
            assert isinstance(name, str)
    
    def test_normalize_features(self):
        """Test feature normalization."""
        # Create test features
        test_features = {
            'pitch_mean': 200.0,
            'energy_mean': 0.15,
            'tempo': 140.0,
            'unknown_feature': 0.5
        }
        
        normalized = self.extractor.normalize_features(test_features)
        
        # Check that all features are present
        for key in test_features:
            assert key in normalized
        
        # Check that unknown features are unchanged
        assert normalized['unknown_feature'] == 0.5
    
    def test_normalize_features_with_params(self):
        """Test feature normalization with custom parameters."""
        test_features = {'pitch_mean': 200.0}
        
        custom_params = {
            'pitch_mean': {'mean': 150.0, 'std': 50.0}
        }
        
        normalized = self.extractor.normalize_features(test_features, custom_params)
        
        # Check z-score normalization: (200 - 150) / 50 = 1.0
        assert abs(normalized['pitch_mean'] - 1.0) < 0.001
    
    def test_edge_cases(self):
        """Test edge cases."""
        # Test with very short audio
        short_audio = np.random.randn(100)
        features = self.extractor.extract_features(short_audio)
        
        assert isinstance(features, dict)
        assert len(features) > 0
        
        # Test with empty audio
        empty_audio = np.array([])
        features = self.extractor.extract_features(empty_audio)
        
        assert isinstance(features, dict)
        assert features['duration'] == 0.0
        
        # Test with silence
        silence = np.zeros(22050)
        features = self.extractor.extract_features(silence)
        
        assert isinstance(features, dict)
        assert features['rms_energy'] == 0.0
    
    def test_feature_consistency(self):
        """Test that features are consistent across multiple extractions."""
        features1 = self.extractor.extract_features(self.test_audio)
        features2 = self.extractor.extract_features(self.test_audio)
        
        # Features should be identical for same input
        for key in features1:
            assert key in features2
            assert abs(features1[key] - features2[key]) < 1e-10
    
    def test_feature_ranges(self):
        """Test that features are within reasonable ranges."""
        features = self.extractor.extract_features(self.test_audio)
        
        # Check specific feature ranges
        assert features['duration'] > 0
        assert features['rms_energy'] >= 0
        assert features['zero_crossing_rate'] >= 0
        assert features['tempo'] > 0
        assert features['speech_rate'] >= 0
