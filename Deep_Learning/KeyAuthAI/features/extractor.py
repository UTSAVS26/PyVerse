"""
Feature Extractor Module for KeyAuthAI

This module extracts behavioral features from keystroke dynamics data:
- Dwell time (how long keys are held)
- Flight time (time between key releases and presses)
- N-gram patterns and timing
- Statistical features for machine learning
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import statistics


class FeatureExtractor:
    """Extracts behavioral features from keystroke dynamics data."""
    
    def __init__(self):
        """Initialize the feature extractor."""
        self.feature_names = []
    
    def extract_features(self, session_data: List[Dict]) -> Dict[str, float]:
        """
        Extract all features from a keystroke session.
        
        Args:
            session_data: List of keystroke events
            
        Returns:
            Dictionary of feature names and values
        """
        if not session_data:
            return {}
        
        features = {}
        
        try:
            # Basic timing features
            features.update(self._extract_timing_features(session_data))
            
            # Dwell time features
            features.update(self._extract_dwell_features(session_data))
            
            # Flight time features
            features.update(self._extract_flight_features(session_data))
            
            # N-gram features
            features.update(self._extract_ngram_features(session_data))
            
            # Statistical features
            features.update(self._extract_statistical_features(session_data))
            
            # Pattern features
            features.update(self._extract_pattern_features(session_data))
            
        except Exception as e:
            # Return empty features if extraction fails
            print(f"Warning: Feature extraction failed: {e}")
            return {}
        
        self.feature_names = list(features.keys())
        return features
    
    def _extract_timing_features(self, session_data: List[Dict]) -> Dict[str, float]:
        """Extract basic timing features."""
        features = {}
        
        if not session_data:
            return features
        
        try:
            # Validate that required fields exist
            if not all('timestamp' in event for event in session_data):
                return features
            
            # Total session time
            timestamps = [event['timestamp'] for event in session_data if isinstance(event.get('timestamp'), (int, float))]
            if len(timestamps) >= 2:
                total_time = max(timestamps) - min(timestamps)
                features['total_time'] = float(total_time)
                
                # Average time between events
                intervals = []
                for i in range(1, len(session_data)):
                    if (isinstance(session_data[i].get('timestamp'), (int, float)) and 
                        isinstance(session_data[i-1].get('timestamp'), (int, float))):
                        interval = session_data[i]['timestamp'] - session_data[i-1]['timestamp']
                        intervals.append(interval)
                
                if intervals:
                    features['avg_interval'] = float(np.mean(intervals))
                    features['std_interval'] = float(np.std(intervals))
                    features['min_interval'] = float(np.min(intervals))
                    features['max_interval'] = float(np.max(intervals))
                
                # Typing speed (characters per second)
                press_events = [e for e in session_data if e.get('type') == 'press']
                if press_events and total_time > 0:
                    features['typing_speed_cps'] = float(len(press_events) / total_time)
        except Exception as e:
            print(f"Warning: Timing feature extraction failed: {e}")
        
        return features
    
    def _extract_dwell_features(self, session_data: List[Dict]) -> Dict[str, float]:
        """Extract dwell time features."""
        features = {}
        
        try:
            dwell_times = []
            for e in session_data:
                if (e.get('type') == 'release' and 
                    'dwell_time' in e and 
                    isinstance(e['dwell_time'], (int, float))):
                    dwell_times.append(float(e['dwell_time']))
            
            if dwell_times:
                features['avg_dwell_time'] = float(np.mean(dwell_times))
                features['std_dwell_time'] = float(np.std(dwell_times))
                features['min_dwell_time'] = float(np.min(dwell_times))
                features['max_dwell_time'] = float(np.max(dwell_times))
                features['median_dwell_time'] = float(np.median(dwell_times))
                
                # Dwell time percentiles
                features['dwell_time_25th'] = float(np.percentile(dwell_times, 25))
                features['dwell_time_75th'] = float(np.percentile(dwell_times, 75))
                features['dwell_time_iqr'] = float(features['dwell_time_75th'] - features['dwell_time_25th'])
        except Exception as e:
            print(f"Warning: Dwell feature extraction failed: {e}")
        
        return features
    
    def _extract_flight_features(self, session_data: List[Dict]) -> Dict[str, float]:
        """Extract flight time features."""
        features = {}
        
        try:
            flight_times = []
            for e in session_data:
                if ('flight_time' in e and 
                    isinstance(e['flight_time'], (int, float))):
                    flight_times.append(float(e['flight_time']))
            
            if flight_times:
                features['avg_flight_time'] = float(np.mean(flight_times))
                features['std_flight_time'] = float(np.std(flight_times))
                features['min_flight_time'] = float(np.min(flight_times))
                features['max_flight_time'] = float(np.max(flight_times))
                features['median_flight_time'] = float(np.median(flight_times))
                
                # Flight time percentiles
                features['flight_time_25th'] = float(np.percentile(flight_times, 25))
                features['flight_time_75th'] = float(np.percentile(flight_times, 75))
                features['flight_time_iqr'] = float(features['flight_time_75th'] - features['flight_time_25th'])
        except Exception as e:
            print(f"Warning: Flight feature extraction failed: {e}")
        
        return features
    
    def _extract_ngram_features(self, session_data: List[Dict], max_n: int = 3) -> Dict[str, float]:
        """Extract n-gram timing features."""
        features = {}
        
        try:
            # Get key sequence
            key_sequence = [e['key'] for e in session_data 
                          if e.get('type') == 'press' and e.get('key')]
            
            for n in range(2, min(max_n + 1, len(key_sequence) + 1)):
                ngram_times = []
                
                for i in range(len(key_sequence) - n + 1):
                    ngram = ''.join(key_sequence[i:i+n])
                    
                    # Find corresponding events
                    start_idx = i
                    end_idx = i + n - 1
                    
                    # Find the press events for this n-gram
                    press_events = [e for e in session_data if e.get('type') == 'press']
                    if (start_idx < len(press_events) and end_idx < len(press_events) and
                        isinstance(press_events[start_idx].get('timestamp'), (int, float)) and
                        isinstance(press_events[end_idx].get('timestamp'), (int, float))):
                        start_time = press_events[start_idx]['timestamp']
                        end_time = press_events[end_idx]['timestamp']
                        ngram_time = end_time - start_time
                        ngram_times.append(float(ngram_time))
                
                if ngram_times:
                    features[f'avg_{n}gram_time'] = float(np.mean(ngram_times))
                    features[f'std_{n}gram_time'] = float(np.std(ngram_times))
                    features[f'min_{n}gram_time'] = float(np.min(ngram_times))
                    features[f'max_{n}gram_time'] = float(np.max(ngram_times))
        except Exception as e:
            print(f"Warning: N-gram feature extraction failed: {e}")
        
        return features
    
    def _extract_statistical_features(self, session_data: List[Dict]) -> Dict[str, float]:
        """Extract statistical features."""
        features = {}
        
        try:
            # Coefficient of variation for dwell and flight times
            dwell_times = [float(e['dwell_time']) for e in session_data 
                          if e.get('type') == 'release' and 
                          'dwell_time' in e and 
                          isinstance(e['dwell_time'], (int, float))]
            flight_times = [float(e['flight_time']) for e in session_data 
                           if 'flight_time' in e and 
                           isinstance(e['flight_time'], (int, float))]
            
            if dwell_times and np.mean(dwell_times) > 0:
                features['dwell_cv'] = float(np.std(dwell_times) / np.mean(dwell_times))
            
            if flight_times and np.mean(flight_times) > 0:
                features['flight_cv'] = float(np.std(flight_times) / np.mean(flight_times))
            
            # Skewness and kurtosis
            if len(dwell_times) > 2:
                features['dwell_skewness'] = float(self._calculate_skewness(dwell_times))
                features['dwell_kurtosis'] = float(self._calculate_kurtosis(dwell_times))
            
            if len(flight_times) > 2:
                features['flight_skewness'] = float(self._calculate_skewness(flight_times))
                features['flight_kurtosis'] = float(self._calculate_kurtosis(flight_times))
        except Exception as e:
            print(f"Warning: Statistical feature extraction failed: {e}")
        
        return features
    
    def _extract_pattern_features(self, session_data: List[Dict]) -> Dict[str, float]:
        """Extract pattern-based features."""
        features = {}
        
        try:
            # Key press intervals
            press_events = [e for e in session_data if e.get('type') == 'press']
            press_intervals = []
            
            for i in range(1, len(press_events)):
                if (isinstance(press_events[i].get('timestamp'), (int, float)) and
                    isinstance(press_events[i-1].get('timestamp'), (int, float))):
                    interval = press_events[i]['timestamp'] - press_events[i-1]['timestamp']
                    press_intervals.append(float(interval))
            
            if press_intervals:
                features['avg_press_interval'] = float(np.mean(press_intervals))
                features['std_press_interval'] = float(np.std(press_intervals))
                features['press_interval_cv'] = float(np.std(press_intervals) / np.mean(press_intervals) if np.mean(press_intervals) > 0 else 0)
            
            # Rhythm consistency (variance of intervals)
            if press_intervals:
                features['rhythm_consistency'] = float(1.0 / (1.0 + np.var(press_intervals)))
            
            # Error patterns (backspace usage, etc.)
            backspace_count = sum(1 for e in session_data 
                                if e.get('type') == 'press' and e.get('key') == '\x08')
            total_presses = len([e for e in session_data if e.get('type') == 'press'])
            features['backspace_ratio'] = float(backspace_count / total_presses if total_presses > 0 else 0)
        except Exception as e:
            print(f"Warning: Pattern feature extraction failed: {e}")
        
        return features
    
    def _calculate_skewness(self, data: List[float]) -> float:
        """Calculate skewness of data."""
        if len(data) < 3:
            return 0.0
        
        try:
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0.0
            
            n = len(data)
            skewness = (n / ((n-1) * (n-2))) * np.sum(((data - mean) / std) ** 3)
            return float(skewness)
        except Exception:
            return 0.0
    
    def _calculate_kurtosis(self, data: List[float]) -> float:
        """Calculate kurtosis of data."""
        if len(data) < 4:
            return 0.0
        
        try:
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0.0
            
            n = len(data)
            kurtosis = (n * (n+1) / ((n-1) * (n-2) * (n-3))) * np.sum(((data - mean) / std) ** 4) - (3 * (n-1)**2 / ((n-2) * (n-3)))
            return float(kurtosis)
        except Exception:
            return 0.0
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return self.feature_names
    
    def extract_features_batch(self, sessions_data: List[List[Dict]]) -> List[Dict[str, float]]:
        """
        Extract features from multiple sessions.
        
        Args:
            sessions_data: List of session data lists
            
        Returns:
            List of feature dictionaries
        """
        features_list = []
        
        for session_data in sessions_data:
            features = self.extract_features(session_data)
            features_list.append(features)
        
        return features_list


def extract_features_from_session(session_data: List[Dict]) -> Dict[str, float]:
    """
    Convenience function to extract features from a single session.
    
    Args:
        session_data: List of keystroke events
        
    Returns:
        Dictionary of features
    """
    extractor = FeatureExtractor()
    return extractor.extract_features(session_data)


def extract_features_from_sessions(sessions_data: List[List[Dict]]) -> List[Dict[str, float]]:
    """
    Convenience function to extract features from multiple sessions.
    
    Args:
        sessions_data: List of session data lists
        
    Returns:
        List of feature dictionaries
    """
    extractor = FeatureExtractor()
    return extractor.extract_features_batch(sessions_data)


if __name__ == "__main__":
    # Example usage
    print("KeyAuthAI Feature Extractor")
    print("=" * 30)
    
    # Sample session data
    sample_data = [
        {'type': 'press', 'key': 't', 'timestamp': 0.0, 'relative_time': 0.0},
        {'type': 'release', 'key': 't', 'timestamp': 0.1, 'relative_time': 0.1, 'dwell_time': 0.1},
        {'type': 'press', 'key': 'h', 'timestamp': 0.2, 'relative_time': 0.2},
        {'type': 'release', 'key': 'h', 'timestamp': 0.25, 'relative_time': 0.25, 'dwell_time': 0.05, 'flight_time': 0.1},
        {'type': 'press', 'key': 'e', 'timestamp': 0.3, 'relative_time': 0.3},
        {'type': 'release', 'key': 'e', 'timestamp': 0.35, 'relative_time': 0.35, 'dwell_time': 0.05, 'flight_time': 0.05},
    ]
    
    extractor = FeatureExtractor()
    features = extractor.extract_features(sample_data)
    
    print("Extracted features:")
    for name, value in features.items():
        print(f"  {name}: {value:.4f}") 