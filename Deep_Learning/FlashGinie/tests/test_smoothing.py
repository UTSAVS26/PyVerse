import pytest
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import os
import json

from utils.smoothing import MoodSmoother, AdaptiveSmoother


class TestMoodSmoother:
    """Test cases for MoodSmoother class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.smoother = MoodSmoother(window_size=5, smoothing_method='simple')
        self.sample_moods = ['happy', 'excited', 'calm', 'stressed', 'happy']
        
    def test_initialization(self):
        """Test MoodSmoother initialization."""
        assert self.smoother.window_size == 5
        assert self.smoother.smoothing_method == 'simple'
        assert len(self.smoother.mood_history) == 0
        assert self.smoother.emotion_values == {
            'happy': 1.0, 'excited': 0.8, 'calm': 0.6, 
            'neutral': 0.5, 'sad': 0.3, 'stressed': 0.2, 'angry': 0.0
        }
        
    def test_initialization_invalid_method(self):
        """Test initialization with invalid smoothing method."""
        with pytest.raises(ValueError, match="Invalid smoothing method"):
            MoodSmoother(smoothing_method='invalid')
            
    def test_add_mood_prediction(self):
        """Test adding mood prediction to history."""
        self.smoother.add_mood_prediction('happy', confidence=0.8)
        
        assert len(self.smoother.mood_history) == 1
        assert self.smoother.mood_history[0]['mood'] == 'happy'
        assert self.smoother.mood_history[0]['confidence'] == 0.8
        assert 'timestamp' in self.smoother.mood_history[0]
        
    def test_get_smoothed_mood_simple(self):
        """Test simple smoothing method."""
        # Add some mood predictions
        for mood in self.sample_moods:
            self.smoother.add_mood_prediction(mood, confidence=0.8)
            
        smoothed_mood = self.smoother.get_smoothed_mood()
        
        assert isinstance(smoothed_mood, dict)
        assert 'mood' in smoothed_mood
        assert 'confidence' in smoothed_mood
        assert smoothed_mood['mood'] in self.smoother.emotion_values.keys()
        assert 0 <= smoothed_mood['confidence'] <= 1
        
    def test_get_smoothed_mood_exponential(self):
        """Test exponential smoothing method."""
        smoother = MoodSmoother(window_size=5, smoothing_method='exponential', alpha=0.3)
        
        for mood in self.sample_moods:
            smoother.add_mood_prediction(mood, confidence=0.8)
            
        smoothed_mood = smoother.get_smoothed_mood()
        
        assert isinstance(smoothed_mood, dict)
        assert 'mood' in smoothed_mood
        assert 'confidence' in smoothed_mood
        
    def test_get_smoothed_mood_weighted(self):
        """Test weighted smoothing method."""
        smoother = MoodSmoother(window_size=5, smoothing_method='weighted')
        
        for mood in self.sample_moods:
            smoother.add_mood_prediction(mood, confidence=0.8)
            
        smoothed_mood = smoother.get_smoothed_mood()
        
        assert isinstance(smoothed_mood, dict)
        assert 'mood' in smoothed_mood
        assert 'confidence' in smoothed_mood
        
    def test_get_smoothed_mood_empty_history(self):
        """Test getting smoothed mood with empty history."""
        smoothed_mood = self.smoother.get_smoothed_mood()
        
        assert smoothed_mood['mood'] == 'neutral'
        assert smoothed_mood['confidence'] == 0.5
        
    def test_simple_smoothing(self):
        """Test simple smoothing calculation."""
        moods = ['happy', 'excited', 'calm']
        confidences = [0.8, 0.9, 0.7]
        
        for mood, conf in zip(moods, confidences):
            self.smoother.add_mood_prediction(mood, confidence=conf)
            
        smoothed = self.smoother._simple_smoothing()
        
        assert isinstance(smoothed, dict)
        assert 'mood' in smoothed
        assert 'confidence' in smoothed
        
    def test_exponential_smoothing(self):
        """Test exponential smoothing calculation."""
        smoother = MoodSmoother(window_size=5, smoothing_method='exponential', alpha=0.3)
        
        moods = ['happy', 'excited', 'calm']
        confidences = [0.8, 0.9, 0.7]
        
        for mood, conf in zip(moods, confidences):
            smoother.add_mood_prediction(mood, confidence=conf)
            
        smoothed = smoother._exponential_smoothing()
        
        assert isinstance(smoothed, dict)
        assert 'mood' in smoothed
        assert 'confidence' in smoothed
        
    def test_weighted_smoothing(self):
        """Test weighted smoothing calculation."""
        smoother = MoodSmoother(window_size=5, smoothing_method='weighted')
        
        moods = ['happy', 'excited', 'calm']
        confidences = [0.8, 0.9, 0.7]
        
        for mood, conf in zip(moods, confidences):
            smoother.add_mood_prediction(mood, confidence=conf)
            
        smoothed = smoother._weighted_smoothing()
        
        assert isinstance(smoothed, dict)
        assert 'mood' in smoothed
        assert 'confidence' in smoothed
        
    def test_emotions_to_values(self):
        """Test conversion of emotions to numerical values."""
        values = self.smoother._emotions_to_values(['happy', 'excited', 'calm'])
        
        assert isinstance(values, list)
        assert len(values) == 3
        assert all(isinstance(v, float) for v in values)
        assert all(0 <= v <= 1 for v in values)
        
    def test_value_to_emotion(self):
        """Test conversion of numerical value to emotion."""
        emotion = self.smoother._value_to_emotion(0.8)
        
        assert isinstance(emotion, str)
        assert emotion in self.smoother.emotion_values.keys()
        
    def test_get_mood_trend(self):
        """Test mood trend analysis."""
        # Add mood predictions with a clear trend
        moods = ['sad', 'neutral', 'calm', 'excited', 'happy']
        for mood in moods:
            self.smoother.add_mood_prediction(mood, confidence=0.8)
            
        trend = self.smoother.get_mood_trend()
        
        assert isinstance(trend, str)
        assert trend in ['improving', 'declining', 'stable']
        
    def test_get_mood_stability(self):
        """Test mood stability calculation."""
        # Add consistent mood predictions
        for _ in range(5):
            self.smoother.add_mood_prediction('happy', confidence=0.8)
            
        stability = self.smoother.get_mood_stability()
        
        assert isinstance(stability, float)
        assert 0 <= stability <= 1
        
    def test_clear_history(self):
        """Test clearing mood history."""
        # Add some predictions
        for mood in self.sample_moods:
            self.smoother.add_mood_prediction(mood, confidence=0.8)
            
        self.smoother.clear_history()
        
        assert len(self.smoother.mood_history) == 0
        
    def test_get_smoothing_stats(self):
        """Test getting smoothing statistics."""
        # Add some predictions
        for mood in self.sample_moods:
            self.smoother.add_mood_prediction(mood, confidence=0.8)
            
        stats = self.smoother.get_smoothing_stats()
        
        assert 'total_predictions' in stats
        assert 'current_window_size' in stats
        assert 'smoothing_method' in stats
        assert 'average_confidence' in stats
        assert stats['total_predictions'] == 5
        assert stats['smoothing_method'] == 'simple'
        
    def test_edge_cases(self):
        """Test edge cases for mood smoothing."""
        # Test with single prediction
        self.smoother.add_mood_prediction('happy', confidence=0.9)
        smoothed = self.smoother.get_smoothed_mood()
        
        assert smoothed['mood'] == 'happy'
        assert smoothed['confidence'] == 0.9
        
        # Test with very low confidence
        self.smoother.clear_history()
        self.smoother.add_mood_prediction('happy', confidence=0.1)
        smoothed = self.smoother.get_smoothed_mood()
        
        assert smoothed['confidence'] == 0.1
        
        # Test with very high confidence
        self.smoother.clear_history()
        self.smoother.add_mood_prediction('excited', confidence=0.99)
        smoothed = self.smoother.get_smoothed_mood()
        
        assert smoothed['confidence'] == 0.99
        
    def test_window_size_limits(self):
        """Test behavior with different window sizes."""
        # Test with small window
        small_smoother = MoodSmoother(window_size=2)
        small_smoother.add_mood_prediction('happy', confidence=0.8)
        small_smoother.add_mood_prediction('sad', confidence=0.7)
        small_smoother.add_mood_prediction('excited', confidence=0.9)
        
        smoothed = small_smoother.get_smoothed_mood()
        assert isinstance(smoothed, dict)
        
        # Test with large window
        large_smoother = MoodSmoother(window_size=20)
        for mood in self.sample_moods * 4:  # 20 predictions
            large_smoother.add_mood_prediction(mood, confidence=0.8)
            
        smoothed = large_smoother.get_smoothed_mood()
        assert isinstance(smoothed, dict)


class TestAdaptiveSmoother:
    """Test cases for AdaptiveSmoother class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.adaptive_smoother = AdaptiveSmoother(
            min_window_size=3,
            max_window_size=10,
            stability_threshold=0.7
        )
        
    def test_initialization(self):
        """Test AdaptiveSmoother initialization."""
        assert self.adaptive_smoother.min_window_size == 3
        assert self.adaptive_smoother.max_window_size == 10
        assert self.adaptive_smoother.stability_threshold == 0.7
        assert self.adaptive_smoother.current_window_size == 10
        assert len(self.adaptive_smoother.mood_history) == 0
        
    def test_add_mood_prediction(self):
        """Test adding mood prediction with adaptive window adjustment."""
        # Add predictions to trigger window size adjustment
        for mood in ['happy', 'excited', 'calm', 'stressed', 'happy']:
            self.adaptive_smoother.add_mood_prediction(mood, confidence=0.8)
            
        assert len(self.adaptive_smoother.mood_history) == 5
        assert self.adaptive_smoother.current_window_size >= 3
        assert self.adaptive_smoother.current_window_size <= 10
        
    def test_get_smoothed_mood(self):
        """Test getting smoothed mood with adaptive smoothing."""
        # Add some predictions
        for mood in ['happy', 'excited', 'calm', 'stressed', 'happy']:
            self.adaptive_smoother.add_mood_prediction(mood, confidence=0.8)
            
        smoothed_mood = self.adaptive_smoother.get_smoothed_mood()
        
        assert isinstance(smoothed_mood, dict)
        assert 'mood' in smoothed_mood
        assert 'confidence' in smoothed_mood
        assert smoothed_mood['mood'] in self.adaptive_smoother.emotion_values.keys()
        
    def test_adjust_window_size_stable_mood(self):
        """Test window size adjustment for stable mood."""
        # Add consistent predictions (stable mood)
        for _ in range(5):
            self.adaptive_smoother.add_mood_prediction('happy', confidence=0.8)
            
        initial_window = self.adaptive_smoother.current_window_size
        self.adaptive_smoother._adjust_window_size()
        
        # Window size should increase for stable mood
        assert self.adaptive_smoother.current_window_size >= initial_window
        
    def test_adjust_window_size_unstable_mood(self):
        """Test window size adjustment for unstable mood."""
        # Add varying predictions (unstable mood)
        moods = ['happy', 'sad', 'excited', 'stressed', 'calm']
        for mood in moods:
            self.adaptive_smoother.add_mood_prediction(mood, confidence=0.8)
            
        initial_window = self.adaptive_smoother.current_window_size
        self.adaptive_smoother._adjust_window_size()
        
        # Window size should decrease for unstable mood
        assert self.adaptive_smoother.current_window_size <= initial_window
        
    def test_window_size_limits(self):
        """Test that window size stays within limits."""
        # Force very stable mood to increase window size
        for _ in range(20):
            self.adaptive_smoother.add_mood_prediction('happy', confidence=0.9)
            
        assert self.adaptive_smoother.current_window_size <= self.adaptive_smoother.max_window_size
        
        # Force very unstable mood to decrease window size
        self.adaptive_smoother.clear_history()
        moods = ['happy', 'sad', 'excited', 'stressed', 'calm', 'angry', 'neutral']
        for mood in moods:
            self.adaptive_smoother.add_mood_prediction(mood, confidence=0.8)
            
        assert self.adaptive_smoother.current_window_size >= self.adaptive_smoother.min_window_size
        
    def test_get_adaptive_stats(self):
        """Test getting adaptive smoothing statistics."""
        # Add some predictions
        for mood in ['happy', 'excited', 'calm', 'stressed', 'happy']:
            self.adaptive_smoother.add_mood_prediction(mood, confidence=0.8)
            
        stats = self.adaptive_smoother.get_adaptive_stats()
        
        assert 'current_window_size' in stats
        assert 'min_window_size' in stats
        assert 'max_window_size' in stats
        assert 'stability_threshold' in stats
        assert 'mood_stability' in stats
        assert 'window_adjustments' in stats
        
    def test_clear_history(self):
        """Test clearing history and resetting window size."""
        # Add some predictions
        for mood in ['happy', 'excited', 'calm']:
            self.adaptive_smoother.add_mood_prediction(mood, confidence=0.8)
            
        self.adaptive_smoother.clear_history()
        
        assert self.adaptive_smoother.mood_history == []
        assert self.adaptive_smoother.current_window_size == self.adaptive_smoother.min_window_size
        
    def test_edge_cases_adaptive(self):
        """Test edge cases for adaptive smoothing."""
        # Test with single prediction
        self.adaptive_smoother.add_mood_prediction('happy', confidence=0.9)
        smoothed = self.adaptive_smoother.get_smoothed_mood()
        
        assert smoothed['mood'] == 'happy'
        assert smoothed['confidence'] == 0.9
        
        # Test with very low confidence
        self.adaptive_smoother.clear_history()
        self.adaptive_smoother.add_mood_prediction('happy', confidence=0.1)
        smoothed = self.adaptive_smoother.get_smoothed_mood()
        
        assert smoothed['confidence'] == 0.1
        
        # Test with empty history
        self.adaptive_smoother.clear_history()
        smoothed = self.adaptive_smoother.get_smoothed_mood()
        
        assert smoothed['mood'] == 'neutral'
        assert smoothed['confidence'] == 0.5
        
    def test_inheritance_from_mood_smoother(self):
        """Test that AdaptiveSmoother inherits properly from MoodSmoother."""
        # Test that basic MoodSmoother methods work
        self.adaptive_smoother.add_mood_prediction('happy', confidence=0.8)
        self.adaptive_smoother.add_mood_prediction('excited', confidence=0.9)
        
        # Test inherited methods
        trend = self.adaptive_smoother.get_mood_trend()
        stability = self.adaptive_smoother.get_mood_stability()
        stats = self.adaptive_smoother.get_smoothing_stats()
        
        assert isinstance(trend, str)
        assert isinstance(stability, float)
        assert isinstance(stats, dict)
        
    def test_adaptive_behavior_consistency(self):
        """Test that adaptive behavior is consistent across multiple calls."""
        # Add predictions and get smoothed mood multiple times
        moods = ['happy', 'excited', 'calm', 'stressed', 'happy']
        
        results = []
        for mood in moods:
            self.adaptive_smoother.add_mood_prediction(mood, confidence=0.8)
            results.append(self.adaptive_smoother.get_smoothed_mood())
            
        # All results should be valid
        assert all(isinstance(r, dict) for r in results)
        assert all('mood' in r and 'confidence' in r for r in results)
        
        # Window size should be adjusted appropriately
        assert self.adaptive_smoother.current_window_size >= self.adaptive_smoother.min_window_size
        assert self.adaptive_smoother.current_window_size <= self.adaptive_smoother.max_window_size
