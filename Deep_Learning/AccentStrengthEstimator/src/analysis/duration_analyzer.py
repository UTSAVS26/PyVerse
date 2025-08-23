"""
Duration and rhythm analysis functionality.
"""

import numpy as np
from typing import List, Dict, Tuple, Any
from scipy.stats import pearsonr
from scipy.signal import find_peaks


class DurationAnalyzer:
    """Analyzes speech duration, rhythm, and timing patterns."""
    
    def __init__(self, sample_rate: int = 22050):
        """
        Initialize the duration analyzer.
        
        Args:
            sample_rate: Sampling rate in Hz
        """
        self.sample_rate = sample_rate
        
    def extract_duration_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract duration-related features from audio.
        
        Args:
            audio: Audio data
            
        Returns:
            dict: Dictionary containing duration features
        """
        if len(audio) == 0:
            return {}
        
        features = {}
        
        # Basic duration features
        features['total_duration'] = len(audio) / self.sample_rate
        features['speech_rate'] = self._estimate_speech_rate(audio)
        features['pause_ratio'] = self._compute_pause_ratio(audio)
        features['syllable_rate'] = self._estimate_syllable_rate(audio)
        
        # Rhythm features
        rhythm_features = self._analyze_rhythm(audio)
        features.update(rhythm_features)
        
        # Energy envelope features
        energy_features = self._analyze_energy_envelope(audio)
        features.update(energy_features)
        
        return features
    
    def _estimate_speech_rate(self, audio: np.ndarray) -> float:
        """
        Estimate speech rate in words per minute.
        
        Args:
            audio: Audio data
            
        Returns:
            float: Estimated speech rate
        """
        # Simple estimation based on energy peaks
        energy = np.abs(audio)
        threshold = np.mean(energy) + 0.5 * np.std(energy)
        
        # Find peaks above threshold
        peaks, _ = find_peaks(energy, height=threshold, distance=int(0.1 * self.sample_rate))
        
        # Estimate words based on peaks (rough approximation)
        num_words = len(peaks) * 0.8  # Not all peaks are words
        duration_minutes = len(audio) / (self.sample_rate * 60)
        
        if duration_minutes > 0:
            return num_words / duration_minutes
        return 0.0
    
    def _compute_pause_ratio(self, audio: np.ndarray) -> float:
        """
        Compute the ratio of silence/pause time to total time.
        
        Args:
            audio: Audio data
            
        Returns:
            float: Pause ratio between 0 and 1
        """
        # Compute energy envelope
        window_size = int(0.025 * self.sample_rate)  # 25ms windows
        energy = []
        
        for i in range(0, len(audio), window_size):
            window = audio[i:i + window_size]
            if len(window) > 0:
                energy.append(np.mean(np.abs(window)))
        
        energy = np.array(energy)
        
        # Find silence threshold
        threshold = np.percentile(energy, 20)  # Bottom 20% is silence
        
        # Count silent frames
        silent_frames = np.sum(energy < threshold)
        total_frames = len(energy)
        
        if total_frames > 0:
            return silent_frames / total_frames
        return 0.0
    
    def _estimate_syllable_rate(self, audio: np.ndarray) -> float:
        """
        Estimate syllable rate in syllables per second.
        
        Args:
            audio: Audio data
            
        Returns:
            float: Estimated syllable rate
        """
        # Use zero-crossing rate as a proxy for syllable rate
        zero_crossings = np.sum(np.diff(np.sign(audio)) != 0)
        duration = len(audio) / self.sample_rate
        
        if duration > 0:
            # Rough conversion: zero crossings to syllables
            syllable_rate = zero_crossings / (duration * 10)  # Empirical factor
            return max(0.5, min(8.0, syllable_rate))  # Reasonable range
        return 0.0
    
    def _analyze_rhythm(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Analyze speech rhythm patterns.
        
        Args:
            audio: Audio data
            
        Returns:
            dict: Rhythm analysis features
        """
        features = {}
        
        # Compute energy envelope
        window_size = int(0.025 * self.sample_rate)
        energy = []
        
        for i in range(0, len(audio), window_size):
            window = audio[i:i + window_size]
            if len(window) > 0:
                energy.append(np.mean(np.abs(window)))
        
        energy = np.array(energy)
        
        if len(energy) < 2:
            return features
        
        # Rhythm regularity (variance of inter-peak intervals)
        peaks, _ = find_peaks(energy, height=np.mean(energy) + 0.5 * np.std(energy))
        
        if len(peaks) > 1:
            intervals = np.diff(peaks)
            features['rhythm_regularity'] = 1.0 / (1.0 + np.var(intervals))
            features['rhythm_speed'] = np.mean(intervals)
        else:
            features['rhythm_regularity'] = 0.0
            features['rhythm_speed'] = 0.0
        
        # Stress-timing vs syllable-timing
        # English is stress-timed, so we expect more variation in syllable duration
        features['stress_timing_score'] = self._compute_stress_timing_score(energy)
        
        return features
    
    def _compute_stress_timing_score(self, energy: np.ndarray) -> float:
        """
        Compute a score indicating how stress-timed the speech is.
        
        Args:
            energy: Energy envelope
            
        Returns:
            float: Stress timing score (0-1, higher is more stress-timed)
        """
        if len(energy) < 4:
            return 0.0
        
        # Find local maxima (stressed syllables)
        peaks, _ = find_peaks(energy, height=np.mean(energy) + 0.3 * np.std(energy))
        
        if len(peaks) < 2:
            return 0.0
        
        # Compute duration variability between stressed syllables
        intervals = np.diff(peaks)
        cv = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else 0
        
        # Higher CV indicates more stress-timed speech
        stress_score = min(1.0, cv / 0.5)  # Normalize
        return stress_score
    
    def _analyze_energy_envelope(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Analyze energy envelope characteristics.
        
        Args:
            audio: Audio data
            
        Returns:
            dict: Energy envelope features
        """
        features = {}
        
        # Compute energy envelope
        window_size = int(0.025 * self.sample_rate)
        energy = []
        
        for i in range(0, len(audio), window_size):
            window = audio[i:i + window_size]
            if len(window) > 0:
                energy.append(np.mean(np.abs(window)))
        
        energy = np.array(energy)
        
        if len(energy) == 0:
            return features
        
        # Energy statistics
        features['energy_mean'] = np.mean(energy)
        features['energy_std'] = np.std(energy)
        features['energy_cv'] = features['energy_std'] / features['energy_mean'] if features['energy_mean'] > 0 else 0
        
        # Energy contour features
        features['energy_slope'] = self._compute_energy_slope(energy)
        features['energy_complexity'] = self._compute_energy_complexity(energy)
        
        return features
    
    def _compute_energy_slope(self, energy: np.ndarray) -> float:
        """
        Compute the overall slope of the energy envelope.
        
        Args:
            energy: Energy envelope
            
        Returns:
            float: Energy slope
        """
        if len(energy) < 2:
            return 0.0
        
        x = np.arange(len(energy))
        slope, _ = np.polyfit(x, energy, 1)
        return slope
    
    def _compute_energy_complexity(self, energy: np.ndarray) -> float:
        """
        Compute the complexity of the energy envelope.
        
        Args:
            energy: Energy envelope
            
        Returns:
            float: Energy complexity measure
        """
        if len(energy) < 3:
            return 0.0
        
        # Compute first derivative
        derivative = np.diff(energy)
        complexity = np.var(derivative)
        return complexity
    
    def compute_duration_similarity(self, ref_features: Dict[str, float], 
                                  user_features: Dict[str, float]) -> float:
        """
        Compute similarity between duration features.
        
        Args:
            ref_features: Reference duration features
            user_features: User duration features
            
        Returns:
            float: Similarity score between 0 and 1
        """
        if not ref_features or not user_features:
            return 0.0
        
        # Define feature weights
        weights = {
            'speech_rate': 0.3,
            'syllable_rate': 0.2,
            'rhythm_regularity': 0.2,
            'stress_timing_score': 0.2,
            'energy_cv': 0.1
        }
        
        total_similarity = 0.0
        total_weight = 0.0
        
        for feature, weight in weights.items():
            if feature in ref_features and feature in user_features:
                ref_val = ref_features[feature]
                user_val = user_features[feature]
                
                # Normalize values to [0, 1] range
                if feature in ['speech_rate', 'syllable_rate']:
                    # Normalize to reasonable ranges
                    ref_norm = min(1.0, ref_val / 200)  # Max 200 wpm
                    user_norm = min(1.0, user_val / 200)
                else:
                    ref_norm = max(0.0, min(1.0, ref_val))
                    user_norm = max(0.0, min(1.0, user_val))
                
                # Compute similarity
                similarity = 1.0 - abs(ref_norm - user_norm)
                total_similarity += similarity * weight
                total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return total_similarity / total_weight
    
    def analyze_timing_patterns(self, audio: np.ndarray, text: str) -> Dict[str, Any]:
        """
        Analyze timing patterns for a given text.
        
        Args:
            audio: Audio data
            text: Corresponding text
            
        Returns:
            dict: Timing analysis results
        """
        analysis = {
            'speech_rate_category': 'normal',
            'rhythm_type': 'stress_timed',
            'timing_consistency': 0.0,
            'pause_pattern': 'normal',
            'overall_timing_score': 0.0
        }
        
        features = self.extract_duration_features(audio)
        
        if not features:
            return analysis
        
        # Analyze speech rate
        speech_rate = features.get('speech_rate', 0)
        if speech_rate < 120:
            analysis['speech_rate_category'] = 'slow'
        elif speech_rate > 180:
            analysis['speech_rate_category'] = 'fast'
        else:
            analysis['speech_rate_category'] = 'normal'
        
        # Analyze rhythm type
        stress_score = features.get('stress_timing_score', 0)
        if stress_score > 0.6:
            analysis['rhythm_type'] = 'stress_timed'
        elif stress_score < 0.3:
            analysis['rhythm_type'] = 'syllable_timed'
        else:
            analysis['rhythm_type'] = 'mixed'
        
        # Analyze timing consistency
        rhythm_regularity = features.get('rhythm_regularity', 0)
        analysis['timing_consistency'] = rhythm_regularity
        
        # Analyze pause pattern
        pause_ratio = features.get('pause_ratio', 0)
        if pause_ratio > 0.4:
            analysis['pause_pattern'] = 'many_pauses'
        elif pause_ratio < 0.1:
            analysis['pause_pattern'] = 'few_pauses'
        else:
            analysis['pause_pattern'] = 'normal'
        
        # Overall timing score
        analysis['overall_timing_score'] = (
            rhythm_regularity * 0.4 +
            (1.0 - abs(pause_ratio - 0.2)) * 0.3 +  # Optimal pause ratio around 0.2
            stress_score * 0.3
        )
        
        return analysis
    
    def get_timing_feedback(self, ref_analysis: Dict[str, Any],
                            user_analysis: Dict[str, Any]) -> List[str]:
        """
        Generate feedback for timing patterns.
        
        Args:
            ref_analysis: Reference timing analysis
            user_analysis: User timing analysis
            
        Returns:
            list: List of feedback messages
        """
        feedback = []
        
        # Speech rate feedback
        if ref_analysis['speech_rate_category'] != user_analysis['speech_rate_category']:
            if user_analysis['speech_rate_category'] == 'slow':
                feedback.append("Try speaking a bit faster")
            elif user_analysis['speech_rate_category'] == 'fast':
                feedback.append("Try speaking a bit slower")
        
        # Rhythm feedback
        if ref_analysis['rhythm_type'] != user_analysis['rhythm_type']:
            if ref_analysis['rhythm_type'] == 'stress_timed':
                feedback.append("English is stress-timed - vary syllable duration more")
            else:
                feedback.append("Try to maintain more consistent syllable timing")
        
        # Timing consistency feedback
        user_consistency = user_analysis['timing_consistency']
        if user_consistency < 0.5:
            feedback.append("Try to maintain more consistent rhythm")
        
        # Pause pattern feedback
        if ref_analysis['pause_pattern'] != user_analysis['pause_pattern']:
            if user_analysis['pause_pattern'] == 'many_pauses':
                feedback.append("Try to reduce the number of pauses")
            elif user_analysis['pause_pattern'] == 'few_pauses':
                feedback.append("Try to add more natural pauses")
        
        return feedback
