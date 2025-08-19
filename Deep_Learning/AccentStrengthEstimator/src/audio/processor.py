"""
Audio processing and feature extraction functionality.
"""

import librosa
import numpy as np
from typing import Tuple, Dict, List
import scipy.signal as signal
from scipy.stats import pearsonr


class AudioProcessor:
    """Handles audio processing and feature extraction."""
    
    def __init__(self, sample_rate: int = 22050):
        """
        Initialize the audio processor.
        
        Args:
            sample_rate: Sampling rate in Hz (default: 22050)
        """
        self.sample_rate = sample_rate
        
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file and return audio data and sample rate.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            tuple: (audio_data, sample_rate)
        """
        try:
            audio, sr = librosa.load(file_path, sr=self.sample_rate)
            return audio, sr
        except Exception as e:
            print(f"Error loading audio file: {e}")
            return np.array([]), self.sample_rate
    
    def extract_pitch_contour(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract pitch contour from audio using librosa.
        
        Args:
            audio: Audio data
            
        Returns:
            numpy.ndarray: Pitch contour values
        """
        try:
            # Extract pitch using librosa
            pitches, magnitudes = librosa.piptrack(
                y=audio, 
                sr=self.sample_rate,
                hop_length=512,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7')
            )
            
            # Get the pitch values with highest magnitude at each time step
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
                else:
                    pitch_values.append(0)
            
            return np.array(pitch_values)
        except Exception as e:
            print(f"Error extracting pitch contour: {e}")
            return np.array([])
    
    def extract_mfcc(self, audio: np.ndarray, n_mfcc: int = 13) -> np.ndarray:
        """
        Extract MFCC features from audio.
        
        Args:
            audio: Audio data
            n_mfcc: Number of MFCC coefficients
            
        Returns:
            numpy.ndarray: MFCC features
        """
        try:
            mfcc = librosa.feature.mfcc(
                y=audio, 
                sr=self.sample_rate, 
                n_mfcc=n_mfcc
            )
            return mfcc
        except Exception as e:
            print(f"Error extracting MFCC: {e}")
            return np.array([])
    
    def extract_spectral_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract various spectral features from audio.
        
        Args:
            audio: Audio data
            
        Returns:
            dict: Dictionary containing spectral features
        """
        try:
            features = {}
            
            # Spectral centroid
            features['spectral_centroid'] = librosa.feature.spectral_centroid(
                y=audio, sr=self.sample_rate
            )[0]
            
            # Spectral bandwidth
            features['spectral_bandwidth'] = librosa.feature.spectral_bandwidth(
                y=audio, sr=self.sample_rate
            )[0]
            
            # Spectral rolloff
            features['spectral_rolloff'] = librosa.feature.spectral_rolloff(
                y=audio, sr=self.sample_rate
            )[0]
            
            # Zero crossing rate
            features['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(audio)[0]
            
            return features
        except Exception as e:
            print(f"Error extracting spectral features: {e}")
            return {}
    
    def extract_duration_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract duration-related features from audio.
        
        Args:
            audio: Audio data
            
        Returns:
            dict: Dictionary containing duration features
        """
        try:
            features = {}
            
            # Total duration
            features['duration'] = len(audio) / self.sample_rate
            
            # Speech rate (approximate)
            # Count voiced segments
            voiced_frames = librosa.effects.harmonic(audio)
            features['voiced_ratio'] = np.sum(voiced_frames > 0) / len(voiced_frames)
            
            # Energy envelope
            energy = librosa.feature.rms(y=audio)[0]
            features['energy_mean'] = np.mean(energy)
            features['energy_std'] = np.std(energy)
            
            return features
        except Exception as e:
            print(f"Error extracting duration features: {e}")
            return {}
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to prevent clipping and improve processing.
        
        Args:
            audio: Audio data
            
        Returns:
            numpy.ndarray: Normalized audio data
        """
        try:
            # Normalize to [-1, 1] range
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                return audio / max_val
            return audio
        except Exception as e:
            print(f"Error normalizing audio: {e}")
            return audio
    
    def apply_preemphasis(self, audio: np.ndarray, coef: float = 0.97) -> np.ndarray:
        """
        Apply pre-emphasis filter to audio.
        
        Args:
            audio: Audio data
            coef: Pre-emphasis coefficient
            
        Returns:
            numpy.ndarray: Pre-emphasized audio data
        """
        try:
            return np.append(audio[0], audio[1:] - coef * audio[:-1])
        except Exception as e:
            print(f"Error applying pre-emphasis: {e}")
            return audio
    
    def segment_audio(self, audio: np.ndarray, segment_length: float = 0.025, 
                     hop_length: float = 0.010) -> List[np.ndarray]:
        """
        Segment audio into overlapping frames.
        
        Args:
            audio: Audio data
            segment_length: Length of each segment in seconds
            hop_length: Hop length between segments in seconds
            
        Returns:
            list: List of audio segments
        """
        try:
            segment_samples = int(segment_length * self.sample_rate)
            hop_samples = int(hop_length * self.sample_rate)
            
            segments = []
            for i in range(0, len(audio) - segment_samples + 1, hop_samples):
                segment = audio[i:i + segment_samples]
                segments.append(segment)
            
            return segments
        except Exception as e:
            print(f"Error segmenting audio: {e}")
            return []
    
    def compute_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """
        Compute similarity between two feature vectors using correlation.
        
        Args:
            features1: First feature vector
            features2: Second feature vector
            
        Returns:
            float: Similarity score between 0 and 1
        """
        try:
            # Ensure both arrays have the same length
            min_length = min(len(features1), len(features2))
            if min_length == 0:
                return 0.0
            
            features1_trimmed = features1[:min_length]
            features2_trimmed = features2[:min_length]
            
            # Compute correlation coefficient
            correlation, _ = pearsonr(features1_trimmed, features2_trimmed)
            
            # Convert to similarity score (0 to 1)
            similarity = (correlation + 1) / 2
            return max(0.0, min(1.0, similarity))
        except Exception as e:
            print(f"Error computing similarity: {e}")
            return 0.0
