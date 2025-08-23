"""
Pitch and intonation analysis functionality.
"""

import numpy as np
from typing import List, Dict, Tuple, Any
from scipy.stats import pearsonr
from scipy.signal import savgol_filter


class PitchAnalyzer:
    """Analyzes pitch contours and intonation patterns."""
    
    def __init__(self, sample_rate: int = 22050):
        """
        Initialize the pitch analyzer.
        
        Args:
            sample_rate: Sampling rate in Hz
        """
        self.sample_rate = sample_rate
    
    def extract_pitch_contour(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract pitch contour from audio using librosa.
        
        Args:
            audio: Audio data
            
        Returns:
            numpy.ndarray: Pitch contour values
        """
        try:
            import librosa
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
        
    def extract_pitch_features(self, pitch_contour: np.ndarray) -> Dict[str, float]:
        """
        Extract pitch-related features from a pitch contour.
        
        Args:
            pitch_contour: Pitch contour values
            
        Returns:
            dict: Dictionary containing pitch features
        """
        if len(pitch_contour) == 0:
            return {}
        
        # Remove zero values (unvoiced segments)
        voiced_pitch = pitch_contour[pitch_contour > 0]
        
        if len(voiced_pitch) == 0:
            return {}
        
        features = {}
        
        # Basic statistics
        features['mean_pitch'] = np.mean(voiced_pitch)
        features['std_pitch'] = np.std(voiced_pitch)
        features['min_pitch'] = np.min(voiced_pitch)
        features['max_pitch'] = np.max(voiced_pitch)
        features['pitch_range'] = features['max_pitch'] - features['min_pitch']
        
        # Pitch variation
        features['pitch_variance'] = np.var(voiced_pitch)
        features['pitch_cv'] = features['std_pitch'] / features['mean_pitch'] if features['mean_pitch'] > 0 else 0
        
        # Pitch slope (overall trend)
        if len(voiced_pitch) > 1:
            x = np.arange(len(voiced_pitch))
            slope, _ = np.polyfit(x, voiced_pitch, 1)
            features['pitch_slope'] = slope
        else:
            features['pitch_slope'] = 0
        
        # Pitch contour shape features
        features['pitch_contour_complexity'] = self._compute_contour_complexity(voiced_pitch)
        features['pitch_contour_smoothness'] = self._compute_contour_smoothness(voiced_pitch)
        
        return features
    
    def _compute_contour_complexity(self, pitch_contour: np.ndarray) -> float:
        """
        Compute the complexity of a pitch contour.
        
        Args:
            pitch_contour: Pitch contour values
            
        Returns:
            float: Complexity measure
        """
        if len(pitch_contour) < 2:
            return 0.0
        
        # Compute first and second derivatives
        first_derivative = np.diff(pitch_contour)
        second_derivative = np.diff(first_derivative)
        
        # Complexity is based on the variance of derivatives
        complexity = np.var(first_derivative) + np.var(second_derivative)
        return complexity
    
    def _compute_contour_smoothness(self, pitch_contour: np.ndarray) -> float:
        """
        Compute the smoothness of a pitch contour.
        
        Args:
            pitch_contour: Pitch contour values
            
        Returns:
            float: Smoothness measure (0-1, higher is smoother)
        """
        if len(pitch_contour) < 3:
            return 1.0
        
        # Apply smoothing and compare with original
        try:
            smoothed = savgol_filter(pitch_contour, min(5, len(pitch_contour)), 2)
            mse = np.mean((pitch_contour - smoothed) ** 2)
            max_mse = np.var(pitch_contour)
            
            if max_mse == 0:
                return 1.0
            
            smoothness = 1.0 - (mse / max_mse)
            return max(0.0, min(1.0, smoothness))
        except:
            return 0.5
    
    def compute_pitch_similarity(self, pitch1: np.ndarray, pitch2: np.ndarray) -> float:
        """
        Compute similarity between two pitch contours.
        
        Args:
            pitch1: First pitch contour
            pitch2: Second pitch contour
            
        Returns:
            float: Similarity score between 0 and 1
        """
        if len(pitch1) == 0 or len(pitch2) == 0:
            return 0.0
        
        # Normalize pitch contours
        pitch1_norm = self._normalize_pitch_contour(pitch1)
        pitch2_norm = self._normalize_pitch_contour(pitch2)
        
        # Resample to same length
        min_length = min(len(pitch1_norm), len(pitch2_norm))
        if min_length == 0:
            return 0.0
        
        pitch1_resampled = self._resample_pitch_contour(pitch1_norm, min_length)
        pitch2_resampled = self._resample_pitch_contour(pitch2_norm, min_length)
        
        # Compute correlation
        try:
            correlation, _ = pearsonr(pitch1_resampled, pitch2_resampled)
            if np.isnan(correlation):
                return 0.0
            
            # Convert to similarity score
            similarity = (correlation + 1) / 2
            return max(0.0, min(1.0, similarity))
        except:
            return 0.0
    
    def _normalize_pitch_contour(self, pitch_contour: np.ndarray) -> np.ndarray:
        """
        Normalize pitch contour to zero mean and unit variance.
        
        Args:
            pitch_contour: Pitch contour values
            
        Returns:
            numpy.ndarray: Normalized pitch contour
        """
        if len(pitch_contour) == 0:
            return pitch_contour
        
        # Remove zero values
        voiced_pitch = pitch_contour[pitch_contour > 0]
        
        if len(voiced_pitch) == 0:
            return np.zeros_like(pitch_contour)
        
        # Normalize
        mean_pitch = np.mean(voiced_pitch)
        std_pitch = np.std(voiced_pitch)
        
        if std_pitch == 0:
            return np.zeros_like(pitch_contour)
        
        normalized = (pitch_contour - mean_pitch) / std_pitch
        return normalized
    
    def _resample_pitch_contour(self, pitch_contour: np.ndarray, target_length: int) -> np.ndarray:
        """
        Resample pitch contour to target length using linear interpolation.
        
        Args:
            pitch_contour: Pitch contour values
            target_length: Target length
            
        Returns:
            numpy.ndarray: Resampled pitch contour
        """
        if len(pitch_contour) == target_length:
            return pitch_contour
        
        if len(pitch_contour) == 0:
            return np.zeros(target_length)
        
        # Linear interpolation
        x_old = np.linspace(0, 1, len(pitch_contour))
        x_new = np.linspace(0, 1, target_length)
        
        resampled = np.interp(x_new, x_old, pitch_contour)
        return resampled
    
    def analyze_intonation_pattern(self, pitch_contour: np.ndarray, text: str) -> Dict[str, Any]:
        """
        Analyze intonation pattern for a given text.
        
        Args:
            pitch_contour: Pitch contour values
            text: Corresponding text
            
        Returns:
            dict: Intonation analysis results
        """
        analysis = {
            'overall_trend': 'neutral',
            'final_contour': 'neutral',
            'stress_pattern': [],
            'intonation_type': 'declarative',
            'confidence': 0.0
        }
        
        if len(pitch_contour) == 0:
            return analysis
        
        # Remove zero values
        voiced_pitch = pitch_contour[pitch_contour > 0]
        
        if len(voiced_pitch) < 3:
            return analysis
        
        # Analyze overall trend
        x = np.arange(len(voiced_pitch))
        slope, _ = np.polyfit(x, voiced_pitch, 1)
        
        if slope > 0.1:
            analysis['overall_trend'] = 'rising'
        elif slope < -0.1:
            analysis['overall_trend'] = 'falling'
        else:
            analysis['overall_trend'] = 'neutral'
        
        # Analyze final contour (last 20% of the contour)
        final_start = int(len(voiced_pitch) * 0.8)
        final_pitch = voiced_pitch[final_start:]
        
        if len(final_pitch) > 1:
            final_slope, _ = np.polyfit(np.arange(len(final_pitch)), final_pitch, 1)
            
            if final_slope > 0.05:
                analysis['final_contour'] = 'rising'
            elif final_slope < -0.05:
                analysis['final_contour'] = 'falling'
            else:
                analysis['final_contour'] = 'neutral'
        
        # Determine intonation type based on text and pitch pattern
        text_lower = text.lower().strip()
        
        if text_lower.endswith('?'):
            analysis['intonation_type'] = 'interrogative'
        elif text_lower.endswith('!'):
            analysis['intonation_type'] = 'exclamatory'
        elif analysis['final_contour'] == 'rising' and analysis['overall_trend'] == 'rising':
            analysis['intonation_type'] = 'interrogative'
        else:
            analysis['intonation_type'] = 'declarative'
        
        # Compute confidence based on pitch variation
        pitch_variance = np.var(voiced_pitch)
        analysis['confidence'] = min(1.0, pitch_variance / 1000)  # Normalize
        
        return analysis
    
    def compare_intonation_patterns(self, ref_analysis: Dict[str, Any],
                                   user_analysis: Dict[str, Any]) -> float:
        """
        Compare intonation patterns between reference and user.
        
        Args:
            ref_analysis: Reference intonation analysis
            user_analysis: User intonation analysis
            
        Returns:
            float: Similarity score between 0 and 1
        """
        similarity = 0.0
        total_weight = 0.0
        
        # Compare overall trend
        if ref_analysis['overall_trend'] == user_analysis['overall_trend']:
            similarity += 0.3
        total_weight += 0.3
        
        # Compare final contour
        if ref_analysis['final_contour'] == user_analysis['final_contour']:
            similarity += 0.3
        total_weight += 0.3
        
        # Compare intonation type
        if ref_analysis['intonation_type'] == user_analysis['intonation_type']:
            similarity += 0.4
        total_weight += 0.4
        
        if total_weight == 0:
            return 0.0
        
        return similarity / total_weight
    
    def get_intonation_feedback(self, ref_analysis: Dict[str, Any],
                               user_analysis: Dict[str, Any]) -> List[str]:
        """
        Generate feedback for intonation patterns.
        
        Args:
            ref_analysis: Reference intonation analysis
            user_analysis: User intonation analysis
            
        Returns:
            list: List of feedback messages
        """
        feedback = []
        
        # Check overall trend
        if ref_analysis['overall_trend'] != user_analysis['overall_trend']:
            if ref_analysis['overall_trend'] == 'rising':
                feedback.append("Try to use a rising intonation pattern")
            elif ref_analysis['overall_trend'] == 'falling':
                feedback.append("Try to use a falling intonation pattern")
            else:
                feedback.append("Try to maintain a more neutral intonation")
        
        # Check final contour
        if ref_analysis['final_contour'] != user_analysis['final_contour']:
            if ref_analysis['final_contour'] == 'rising':
                feedback.append("End your sentence with a rising pitch")
            elif ref_analysis['final_contour'] == 'falling':
                feedback.append("End your sentence with a falling pitch")
        
        # Check intonation type
        if ref_analysis['intonation_type'] != user_analysis['intonation_type']:
            if ref_analysis['intonation_type'] == 'interrogative':
                feedback.append("This is a question - use rising intonation")
            elif ref_analysis['intonation_type'] == 'exclamatory':
                feedback.append("This is an exclamation - use more dramatic intonation")
            else:
                feedback.append("This is a statement - use falling intonation")
        
        return feedback
