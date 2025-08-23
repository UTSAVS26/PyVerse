"""
Feature extractor module for VoiceMoodMirror.
Extracts prosodic features like pitch, tempo, energy, and spectral characteristics.
"""

import librosa
import numpy as np
from typing import Dict, List, Tuple, Optional
import scipy.stats as stats


class FeatureExtractor:
    """Extracts prosodic features from audio for emotion analysis."""
    
    def __init__(self, sample_rate: int = 22050):
        """
        Initialize the feature extractor.
        
        Args:
            sample_rate: Audio sample rate in Hz
        """
        self.sample_rate = sample_rate
        
    def extract_features(self, audio_data: np.ndarray) -> Dict[str, float]:
        """
        Extract all prosodic features from audio data.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Basic audio features
        features.update(self._extract_basic_features(audio_data))
        
        # Pitch features
        features.update(self._extract_pitch_features(audio_data))
        
        # Energy features
        features.update(self._extract_energy_features(audio_data))
        
        # Spectral features
        features.update(self._extract_spectral_features(audio_data))
        
        # Tempo and rhythm features
        features.update(self._extract_tempo_features(audio_data))
        
        # Voice quality features
        features.update(self._extract_voice_quality_features(audio_data))
        
        return features
    
    def _extract_basic_features(self, audio_data: np.ndarray) -> Dict[str, float]:
        """Extract basic audio features."""
        features = {}
        
        # Duration
        features['duration'] = len(audio_data) / self.sample_rate
        
        # RMS energy
        features['rms_energy'] = np.sqrt(np.mean(audio_data**2))
        
        # Zero crossing rate
        features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(audio_data))
        
        return features
    
    def _extract_pitch_features(self, audio_data: np.ndarray) -> Dict[str, float]:
        """Extract pitch-related features."""
        features = {}
        
        # Extract pitch using librosa
        pitches, magnitudes = librosa.piptrack(y=audio_data, sr=self.sample_rate)
        
        # Get voiced frames (where pitch is detected)
        voiced_frames = magnitudes > 0.1 * np.max(magnitudes)
        voiced_pitches = pitches[voiced_frames]
        
        if len(voiced_pitches) > 0:
            # Convert to Hz and filter out unrealistic values
            voiced_pitches_hz = voiced_pitches[voiced_pitches > 50]  # Remove very low pitches
            
            if len(voiced_pitches_hz) > 0:
                features['pitch_mean'] = np.mean(voiced_pitches_hz)
                features['pitch_std'] = np.std(voiced_pitches_hz)
                features['pitch_min'] = np.min(voiced_pitches_hz)
                features['pitch_max'] = np.max(voiced_pitches_hz)
                features['pitch_range'] = features['pitch_max'] - features['pitch_min']
                
                # Pitch contour statistics
                features['pitch_slope'] = self._calculate_pitch_slope(voiced_pitches_hz)
                features['pitch_variability'] = features['pitch_std'] / features['pitch_mean'] if features['pitch_mean'] > 0 else 0
            else:
                # No valid pitch detected
                features.update({
                    'pitch_mean': 0, 'pitch_std': 0, 'pitch_min': 0, 
                    'pitch_max': 0, 'pitch_range': 0, 'pitch_slope': 0, 'pitch_variability': 0
                })
        else:
            # No voiced frames
            features.update({
                'pitch_mean': 0, 'pitch_std': 0, 'pitch_min': 0, 
                'pitch_max': 0, 'pitch_range': 0, 'pitch_slope': 0, 'pitch_variability': 0
            })
        
        return features
    
    def _extract_energy_features(self, audio_data: np.ndarray) -> Dict[str, float]:
        """Extract energy-related features."""
        features = {}
        
        # Calculate energy envelope
        hop_length = 512
        frame_length = 2048
        
        # RMS energy over time
        rms = librosa.feature.rms(y=audio_data, frame_length=frame_length, hop_length=hop_length)[0]
        
        features['energy_mean'] = np.mean(rms)
        features['energy_std'] = np.std(rms)
        features['energy_max'] = np.max(rms)
        features['energy_min'] = np.min(rms)
        features['energy_range'] = features['energy_max'] - features['energy_min']
        
        # Energy contour statistics
        features['energy_slope'] = self._calculate_energy_slope(rms)
        features['energy_variability'] = features['energy_std'] / features['energy_mean'] if features['energy_mean'] > 0 else 0
        
        # Energy distribution
        features['energy_skewness'] = stats.skew(rms)
        features['energy_kurtosis'] = stats.kurtosis(rms)
        
        return features
    
    def _extract_spectral_features(self, audio_data: np.ndarray) -> Dict[str, float]:
        """Extract spectral features."""
        features = {}
        
        # Spectral centroid (brightness)
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=self.sample_rate)[0]
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_centroid_std'] = np.std(spectral_centroids)
        
        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=self.sample_rate)[0]
        features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
        features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=self.sample_rate)[0]
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        features['spectral_rolloff_std'] = np.std(spectral_rolloff)
        
        # MFCCs (first 5 coefficients)
        mfccs = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, n_mfcc=13)
        for i in range(5):
            features[f'mfcc_{i+1}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i+1}_std'] = np.std(mfccs[i])
        
        return features
    
    def _extract_tempo_features(self, audio_data: np.ndarray) -> Dict[str, float]:
        """Extract tempo and rhythm features."""
        features = {}
        
        # Tempo estimation
        tempo, _ = librosa.beat.beat_track(y=audio_data, sr=self.sample_rate)
        features['tempo'] = tempo
        
        # Onset strength
        onset_env = librosa.onset.onset_strength(y=audio_data, sr=self.sample_rate)
        features['onset_strength_mean'] = np.mean(onset_env)
        features['onset_strength_std'] = np.std(onset_env)
        
        # Speech rate approximation (using zero crossing rate)
        zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
        features['speech_rate'] = np.mean(zcr) * 100  # Scaled for readability
        
        return features
    
    def _extract_voice_quality_features(self, audio_data: np.ndarray) -> Dict[str, float]:
        """Extract voice quality features."""
        features = {}
        
        # Harmonic-to-noise ratio approximation
        # This is a simplified version - in practice, you'd use more sophisticated methods
        hop_length = 512
        frame_length = 2048
        
        # Calculate harmonic and noise components
        stft = librosa.stft(audio_data, hop_length=hop_length, n_fft=frame_length)
        magnitude = np.abs(stft)
        
        # Simple HNR approximation using spectral centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=self.sample_rate)[0]
        features['voice_quality'] = min(np.mean(spectral_centroids) / 1000, 1.0)  # Normalized and capped at 1.0
        
        # Jitter and shimmer approximation (simplified)
        # In practice, these would be calculated from pitch contour
        features['jitter'] = 0.0  # Placeholder
        features['shimmer'] = 0.0  # Placeholder
        
        return features
    
    def _calculate_pitch_slope(self, pitches: np.ndarray) -> float:
        """Calculate the slope of pitch contour."""
        if len(pitches) < 2:
            return 0.0
        
        x = np.arange(len(pitches))
        slope, _ = np.polyfit(x, pitches, 1)
        return slope
    
    def _calculate_energy_slope(self, energy: np.ndarray) -> float:
        """Calculate the slope of energy contour."""
        if len(energy) < 2:
            return 0.0
        
        x = np.arange(len(energy))
        slope, _ = np.polyfit(x, energy, 1)
        return slope
    
    def extract_features_windowed(self, audio_data: np.ndarray, window_duration: float = 3.0) -> List[Dict[str, float]]:
        """
        Extract features from audio using sliding windows.
        
        Args:
            audio_data: Audio data as numpy array
            window_duration: Duration of each window in seconds
            
        Returns:
            List of feature dictionaries for each window
        """
        window_samples = int(window_duration * self.sample_rate)
        hop_samples = window_samples // 2  # 50% overlap
        
        features_list = []
        
        for start in range(0, len(audio_data) - window_samples, hop_samples):
            end = start + window_samples
            window_audio = audio_data[start:end]
            
            # Apply window function to reduce edge effects
            window_audio = window_audio * np.hanning(len(window_audio))
            
            features = self.extract_features(window_audio)
            features_list.append(features)
        
        return features_list
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names."""
        # Create dummy audio to get feature names
        dummy_audio = np.zeros(self.sample_rate)  # 1 second of silence
        features = self.extract_features(dummy_audio)
        return list(features.keys())
    
    def normalize_features(self, features: Dict[str, float], 
                          normalization_params: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, float]:
        """
        Normalize features using z-score normalization.
        
        Args:
            features: Feature dictionary
            normalization_params: Pre-computed normalization parameters (mean, std)
            
        Returns:
            Normalized feature dictionary
        """
        if normalization_params is None:
            # Use default normalization parameters
            normalization_params = self._get_default_normalization_params()
        
        normalized_features = {}
        
        for feature_name, value in features.items():
            if feature_name in normalization_params:
                mean = normalization_params[feature_name]['mean']
                std = normalization_params[feature_name]['std']
                
                if std > 0:
                    normalized_features[feature_name] = (value - mean) / std
                else:
                    normalized_features[feature_name] = 0.0
            else:
                normalized_features[feature_name] = value
        
        return normalized_features
    
    def _get_default_normalization_params(self) -> Dict[str, Dict[str, float]]:
        """Get default normalization parameters based on typical speech characteristics."""
        # These are approximate values based on typical speech characteristics
        # In practice, you'd compute these from a large dataset
        return {
            'pitch_mean': {'mean': 150.0, 'std': 50.0},
            'pitch_std': {'mean': 20.0, 'std': 15.0},
            'energy_mean': {'mean': 0.1, 'std': 0.05},
            'energy_std': {'mean': 0.05, 'std': 0.03},
            'spectral_centroid_mean': {'mean': 2000.0, 'std': 1000.0},
            'tempo': {'mean': 120.0, 'std': 30.0},
            'speech_rate': {'mean': 5.0, 'std': 2.0},
        }
