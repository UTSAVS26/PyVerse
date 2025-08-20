"""
Smoothing module for VoiceMoodMirror.
Provides temporal smoothing of noisy mood predictions.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import deque
import time


class MoodSmoother:
    """Temporal smoothing for mood predictions to reduce jitter."""
    
    def __init__(self, window_size: int = 5, smoothing_method: str = 'simple'):
        """
        Initialize the mood smoother.
        
        Args:
            window_size: Size of the smoothing window
            smoothing_method: Smoothing method ('simple', 'exponential', 'weighted')
        """
        if smoothing_method not in ['simple', 'exponential', 'weighted']:
            raise ValueError("Invalid smoothing method")
            
        self.window_size = window_size
        self.smoothing_method = smoothing_method
        self.mood_history = deque(maxlen=window_size)
        self.confidence_history = deque(maxlen=window_size)
        self.timestamp_history = deque(maxlen=window_size)
        
        # Emotion mapping for numerical conversion
        self.emotion_values = {
            'happy': 1.0, 'excited': 0.8, 'calm': 0.6, 
            'neutral': 0.5, 'sad': 0.3, 'stressed': 0.2, 'angry': 0.0
        }
        
    def add_mood_prediction(self, emotion: str, confidence: float, 
                          timestamp: Optional[float] = None):
        """
        Add a new mood prediction to the smoothing buffer.
        
        Args:
            emotion: Predicted emotion
            confidence: Prediction confidence
            timestamp: Timestamp (defaults to current time)
        """
        if timestamp is None:
            timestamp = time.time()
        
        self.mood_history.append(emotion)
        self.confidence_history.append(confidence)
        self.timestamp_history.append(timestamp)
    
    def get_smoothed_mood(self) -> Dict[str, any]:
        """
        Get the smoothed mood prediction.
        
        Returns:
            Dictionary with 'mood' and 'confidence' keys
        """
        if len(self.mood_history) == 0:
            return {'mood': 'neutral', 'confidence': 0.5}
        
        if len(self.mood_history) == 1:
            return {'mood': self.mood_history[0], 'confidence': self.confidence_history[0]}
        
        if self.smoothing_method == 'simple':
            emotion, confidence = self._simple_smoothing()
        elif self.smoothing_method == 'exponential':
            emotion, confidence = self._exponential_smoothing()
        elif self.smoothing_method == 'weighted':
            emotion, confidence = self._weighted_smoothing()
        else:
            emotion, confidence = self._simple_smoothing()
            
        return {'mood': emotion, 'confidence': confidence}
    
    def _simple_smoothing(self) -> Tuple[str, float]:
        """Simple moving average smoothing."""
        # Count emotions
        emotion_counts = {}
        for emotion in self.mood_history:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        # Get most common emotion
        smoothed_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]
        
        # Average confidence
        smoothed_confidence = np.mean(list(self.confidence_history))
        
        return smoothed_emotion, smoothed_confidence
    
    def _exponential_smoothing(self, alpha: float = 0.7) -> Tuple[str, float]:
        """
        Exponential smoothing with emotion probability weighting.
        
        Args:
            alpha: Smoothing factor (0 < alpha < 1)
        """
        if len(self.mood_history) < 2:
            return self.mood_history[0], self.confidence_history[0]
        
        # Convert emotions to numerical values for smoothing
        emotion_values = self._emotions_to_values(list(self.mood_history))
        
        # Apply exponential smoothing
        smoothed_value = emotion_values[0]
        for i in range(1, len(emotion_values)):
            smoothed_value = alpha * emotion_values[i] + (1 - alpha) * smoothed_value
        
        # Convert back to emotion
        smoothed_emotion = self._value_to_emotion(smoothed_value)
        
        # Smooth confidence
        smoothed_confidence = self.confidence_history[0]
        for i in range(1, len(self.confidence_history)):
            smoothed_confidence = alpha * self.confidence_history[i] + (1 - alpha) * smoothed_confidence
        
        return smoothed_emotion, smoothed_confidence
    
    def _weighted_smoothing(self) -> Tuple[str, float]:
        """Weighted smoothing based on confidence and recency."""
        if len(self.mood_history) == 0:
            return 'neutral', 0.0
        
        # Calculate weights based on confidence and recency
        weights = []
        for i, confidence in enumerate(self.confidence_history):
            # Weight decreases with age and increases with confidence
            recency_weight = 1.0 / (i + 1)
            confidence_weight = confidence
            total_weight = recency_weight * confidence_weight
            weights.append(total_weight)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(weights)] * len(weights)
        
        # Weighted emotion selection
        emotion_scores = {}
        for i, emotion in enumerate(self.mood_history):
            if emotion not in emotion_scores:
                emotion_scores[emotion] = 0.0
            emotion_scores[emotion] += weights[i]
        
        smoothed_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
        
        # Weighted average confidence
        smoothed_confidence = sum(w * c for w, c in zip(weights, self.confidence_history))
        
        return smoothed_emotion, smoothed_confidence
    
    def _emotions_to_values(self, emotions: List[str]) -> List[float]:
        """Convert emotions to numerical values for smoothing."""
        emotion_map = {
            'happy': 5.0,
            'excited': 4.5,
            'calm': 3.0,
            'neutral': 2.5,
            'tired': 1.5,
            'sad': 1.0,
            'angry': 0.5
        }
        
        return [emotion_map.get(emotion, 2.5) for emotion in emotions]
    
    def _value_to_emotion(self, value: float) -> str:
        """Convert numerical value back to emotion."""
        emotion_ranges = [
            (4.5, 5.0, 'happy'),
            (4.0, 4.5, 'excited'),
            (2.5, 4.0, 'calm'),
            (2.0, 2.5, 'neutral'),
            (1.5, 2.0, 'tired'),
            (1.0, 1.5, 'sad'),
            (0.0, 1.0, 'angry')
        ]
        
        for min_val, max_val, emotion in emotion_ranges:
            if min_val <= value <= max_val:
                return emotion
        
        return 'neutral'
    
    def get_mood_trend(self) -> str:
        """
        Analyze mood trend over the smoothing window.
        
        Returns:
            Trend description ('improving', 'declining', 'stable')
        """
        if len(self.mood_history) < 3:
            return 'stable'
        
        # Convert emotions to values
        values = self._emotions_to_values(list(self.mood_history))
        
        # Calculate trend
        if len(values) >= 2:
            slope = (values[-1] - values[0]) / len(values)
            
            if slope > 0.5:
                return 'improving'
            elif slope < -0.5:
                return 'declining'
            else:
                return 'stable'
        
        return 'stable'
    
    def get_mood_stability(self) -> float:
        """
        Calculate mood stability (inverse of variability).
        
        Returns:
            Stability score (0.0 to 1.0, higher is more stable)
        """
        if len(self.mood_history) < 2:
            return 1.0
        
        # Calculate variability in emotions
        emotion_counts = {}
        for emotion in self.mood_history:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        # Stability is inversely proportional to number of different emotions
        num_different_emotions = len(emotion_counts)
        max_possible_emotions = len(self.mood_history)
        
        stability = 1.0 - (num_different_emotions - 1) / (max_possible_emotions - 1)
        return max(0.0, min(1.0, stability))
    
    def clear_history(self):
        """Clear the smoothing history."""
        self.mood_history.clear()
        self.confidence_history.clear()
        self.timestamp_history.clear()
    
    def get_smoothing_stats(self) -> Dict:
        """
        Get statistics about the smoothing process.
        
        Returns:
            Dictionary with smoothing statistics
        """
        if len(self.mood_history) == 0:
            return {
                'total_predictions': 0,
                'current_window_size': self.window_size,
                'smoothing_method': self.smoothing_method,
                'average_confidence': 0.0
            }
        
        return {
            'total_predictions': len(self.mood_history),
            'current_window_size': self.window_size,
            'smoothing_method': self.smoothing_method,
            'average_confidence': np.mean(list(self.confidence_history))
        }


class AdaptiveSmoother(MoodSmoother):
    """Adaptive smoothing that adjusts parameters based on mood stability."""
    
    def __init__(self, min_window_size: int = 3, max_window_size: int = 10, stability_threshold: float = 0.7):
        """
        Initialize the adaptive smoother.
        
        Args:
            min_window_size: Minimum smoothing window size
            max_window_size: Maximum smoothing window size
            stability_threshold: Threshold for stability-based window adjustment
        """
        super().__init__(window_size=max_window_size)
        self.min_window_size = min_window_size
        self.max_window_size = max_window_size
        self.stability_threshold = stability_threshold
        self.current_window_size = max_window_size
        
    def add_mood_prediction(self, emotion: str, confidence: float, 
                          timestamp: Optional[float] = None):
        """
        Add a new mood prediction.
        
        Args:
            emotion: Predicted emotion
            confidence: Prediction confidence
            timestamp: Timestamp
        """
        super().add_mood_prediction(emotion, confidence, timestamp)
        self._adjust_window_size()
    
    def get_smoothed_mood(self) -> Dict[str, any]:
        """
        Get the smoothed mood prediction.
        
        Returns:
            Dictionary with 'mood' and 'confidence' keys
        """
        return super().get_smoothed_mood()
    
    def _adjust_window_size(self):
        """Adjust window size based on mood stability."""
        stability = self.get_mood_stability()
        
        if stability < self.stability_threshold:
            # Low stability: use larger window for more smoothing
            new_window_size = min(self.max_window_size, 
                                self.window_size + 1)
        else:
            # High stability: use smaller window for faster response
            new_window_size = max(self.min_window_size, 
                                self.window_size - 1)
        
        if new_window_size != self.window_size:
            # Update window size
            self.window_size = new_window_size
            self.current_window_size = new_window_size
            # Update deque maxlen
            self.mood_history = deque(self.mood_history, maxlen=new_window_size)
            self.confidence_history = deque(self.confidence_history, maxlen=new_window_size)
            self.timestamp_history = deque(self.timestamp_history, maxlen=new_window_size)
    
    def get_adaptive_stats(self) -> Dict:
        """
        Get statistics about the adaptive smoothing.
        
        Returns:
            Dictionary with adaptive smoothing statistics
        """
        stats = self.get_smoothing_stats()
        stats.update({
            'min_window_size': self.min_window_size,
            'max_window_size': self.max_window_size,
            'stability_threshold': self.stability_threshold,
            'mood_stability': self.get_mood_stability(),
            'window_adjustments': self.current_window_size,
            'adaptive': True
        })
        return stats
