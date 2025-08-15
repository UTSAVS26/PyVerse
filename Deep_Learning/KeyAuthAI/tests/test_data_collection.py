"""
Test Cases for KeyAuthAI Data Collection

This module contains comprehensive tests for:
- Keystroke logging functionality
- Feature extraction
- Data validation
- Error handling
"""

import unittest
import tempfile
import os
import sys
import json
import time
from unittest.mock import patch, MagicMock
from typing import List, Dict

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.keystroke_logger import KeystrokeLogger
from features.extractor import FeatureExtractor


class TestKeystrokeLogger(unittest.TestCase):
    """Test cases for keystroke logging functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_file = os.path.join(self.temp_dir, "test_user_data.json")
        self.logger = KeystrokeLogger(self.data_file)
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.data_file):
            os.remove(self.data_file)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)
    
    def test_initialization(self):
        """Test logger initialization."""
        self.assertIsNotNone(self.logger)
        self.assertEqual(self.logger.data_file, self.data_file)
        self.assertFalse(self.logger.is_recording)
        self.assertEqual(self.logger.current_session, [])
    
    def test_load_user_data_empty(self):
        """Test loading user data when file doesn't exist."""
        data = self.logger._load_user_data()
        self.assertEqual(data, {})
    
    def test_load_user_data_existing(self):
        """Test loading existing user data."""
        test_data = {"user1": {"sessions": [], "passphrase": "test"}}
        with open(self.data_file, 'w') as f:
            json.dump(test_data, f)
        
        data = self.logger._load_user_data()
        self.assertEqual(data, test_data)
    
    def test_save_user_data(self):
        """Test saving user data."""
        test_data = {"user1": {"sessions": [], "passphrase": "test"}}
        self.logger.user_data = test_data
        self.logger._save_user_data()
        
        # Verify file was created
        self.assertTrue(os.path.exists(self.data_file))
        
        # Verify data was saved correctly
        with open(self.data_file, 'r') as f:
            saved_data = json.load(f)
        self.assertEqual(saved_data, test_data)
    
    def test_get_key_char(self):
        """Test key character extraction."""
        # Test regular character
        mock_key = MagicMock()
        mock_key.char = 'a'
        result = self.logger._get_key_char(mock_key)
        self.assertEqual(result, 'a')
        
        # Test None character
        mock_key.char = None
        result = self.logger._get_key_char(mock_key)
        self.assertIsNone(result)
    
    def test_process_session_data(self):
        """Test session data processing."""
        # Create mock session data
        self.logger.current_session = [
            {'type': 'press', 'key': 't', 'timestamp': 0.0, 'relative_time': 0.0},
            {'type': 'release', 'key': 't', 'timestamp': 0.1, 'relative_time': 0.1, 'dwell_time': 0.1},
            {'type': 'press', 'key': 'h', 'timestamp': 0.2, 'relative_time': 0.2},
            {'type': 'release', 'key': 'h', 'timestamp': 0.25, 'relative_time': 0.25, 'dwell_time': 0.05, 'flight_time': 0.1}
        ]
        
        processed_data = self.logger._process_session_data()
        
        self.assertEqual(len(processed_data), 4)
        self.assertEqual(processed_data[0]['key'], 't')
        self.assertEqual(processed_data[0]['type'], 'press')
        self.assertEqual(processed_data[1]['dwell_time'], 0.1)
        self.assertEqual(processed_data[3]['flight_time'], 0.1)
    
    def test_list_users_empty(self):
        """Test listing users when none exist."""
        users = self.logger.list_users()
        self.assertEqual(users, [])
    
    def test_list_users_with_data(self):
        """Test listing users when data exists."""
        self.logger.user_data = {
            "user1": {"sessions": []},
            "user2": {"sessions": []}
        }
        users = self.logger.list_users()
        self.assertEqual(set(users), {"user1", "user2"})
    
    def test_get_user_passphrase(self):
        """Test getting user passphrase."""
        self.logger.user_data = {
            "user1": {"passphrase": "test123", "sessions": []}
        }
        passphrase = self.logger.get_user_passphrase("user1")
        self.assertEqual(passphrase, "test123")
        
        # Test non-existent user
        passphrase = self.logger.get_user_passphrase("nonexistent")
        self.assertIsNone(passphrase)
    
    def test_delete_user_data(self):
        """Test deleting user data."""
        self.logger.user_data = {
            "user1": {"sessions": []},
            "user2": {"sessions": []}
        }
        
        self.logger.delete_user_data("user1")
        
        self.assertNotIn("user1", self.logger.user_data)
        self.assertIn("user2", self.logger.user_data)


class TestFeatureExtractor(unittest.TestCase):
    """Test cases for feature extraction functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = FeatureExtractor()
    
    def test_initialization(self):
        """Test feature extractor initialization."""
        self.assertIsNotNone(self.extractor)
        self.assertEqual(self.extractor.feature_names, [])
    
    def test_extract_features_empty_session(self):
        """Test feature extraction with empty session."""
        features = self.extractor.extract_features([])
        self.assertEqual(features, {})
    
    def test_extract_features_basic(self):
        """Test basic feature extraction."""
        session_data = [
            {'type': 'press', 'key': 't', 'timestamp': 0.0, 'relative_time': 0.0},
            {'type': 'release', 'key': 't', 'timestamp': 0.1, 'relative_time': 0.1, 'dwell_time': 0.1},
            {'type': 'press', 'key': 'h', 'timestamp': 0.2, 'relative_time': 0.2},
            {'type': 'release', 'key': 'h', 'timestamp': 0.25, 'relative_time': 0.25, 'dwell_time': 0.05, 'flight_time': 0.1},
            {'type': 'press', 'key': 'e', 'timestamp': 0.3, 'relative_time': 0.3},
            {'type': 'release', 'key': 'e', 'timestamp': 0.35, 'relative_time': 0.35, 'dwell_time': 0.05, 'flight_time': 0.05}
        ]
        
        features = self.extractor.extract_features(session_data)
        
        # Check that features were extracted
        self.assertGreater(len(features), 0)
        self.assertIn('total_time', features)
        self.assertIn('avg_dwell_time', features)
        self.assertIn('avg_flight_time', features)
    
    def test_extract_timing_features(self):
        """Test timing feature extraction."""
        session_data = [
            {'type': 'press', 'key': 't', 'timestamp': 0.0, 'relative_time': 0.0},
            {'type': 'release', 'key': 't', 'timestamp': 0.1, 'relative_time': 0.1, 'dwell_time': 0.1},
            {'type': 'press', 'key': 'h', 'timestamp': 0.2, 'relative_time': 0.2},
            {'type': 'release', 'key': 'h', 'timestamp': 0.25, 'relative_time': 0.25, 'dwell_time': 0.05, 'flight_time': 0.1}
        ]
        
        features = self.extractor._extract_timing_features(session_data)
        
        self.assertIn('total_time', features)
        self.assertIn('avg_interval', features)
        self.assertIn('typing_speed_cps', features)
        self.assertEqual(features['total_time'], 0.25)
    
    def test_extract_dwell_features(self):
        """Test dwell time feature extraction."""
        session_data = [
            {'type': 'press', 'key': 't', 'timestamp': 0.0, 'relative_time': 0.0},
            {'type': 'release', 'key': 't', 'timestamp': 0.1, 'relative_time': 0.1, 'dwell_time': 0.1},
            {'type': 'press', 'key': 'h', 'timestamp': 0.2, 'relative_time': 0.2},
            {'type': 'release', 'key': 'h', 'timestamp': 0.25, 'relative_time': 0.25, 'dwell_time': 0.05, 'flight_time': 0.1}
        ]
        
        features = self.extractor._extract_dwell_features(session_data)
        
        self.assertIn('avg_dwell_time', features)
        self.assertIn('std_dwell_time', features)
        self.assertIn('min_dwell_time', features)
        self.assertIn('max_dwell_time', features)
        # Use approximate comparison for floating point values
        self.assertAlmostEqual(features['avg_dwell_time'], 0.075, places=10)
    
    def test_extract_flight_features(self):
        """Test flight time feature extraction."""
        session_data = [
            {'type': 'press', 'key': 't', 'timestamp': 0.0, 'relative_time': 0.0},
            {'type': 'release', 'key': 't', 'timestamp': 0.1, 'relative_time': 0.1, 'dwell_time': 0.1},
            {'type': 'press', 'key': 'h', 'timestamp': 0.2, 'relative_time': 0.2},
            {'type': 'release', 'key': 'h', 'timestamp': 0.25, 'relative_time': 0.25, 'dwell_time': 0.05, 'flight_time': 0.1}
        ]
        
        features = self.extractor._extract_flight_features(session_data)
        
        self.assertIn('avg_flight_time', features)
        self.assertAlmostEqual(features['avg_flight_time'], 0.1, places=10)
    
    def test_extract_ngram_features(self):
        """Test n-gram feature extraction."""
        session_data = [
            {'type': 'press', 'key': 't', 'timestamp': 0.0, 'relative_time': 0.0},
            {'type': 'press', 'key': 'h', 'timestamp': 0.1, 'relative_time': 0.1},
            {'type': 'press', 'key': 'e', 'timestamp': 0.2, 'relative_time': 0.2}
        ]
        
        features = self.extractor._extract_ngram_features(session_data)
        
        self.assertIn('avg_2gram_time', features)
        self.assertIn('avg_3gram_time', features)
    
    def test_extract_statistical_features(self):
        """Test statistical feature extraction."""
        session_data = [
            {'type': 'press', 'key': 't', 'timestamp': 0.0, 'relative_time': 0.0},
            {'type': 'release', 'key': 't', 'timestamp': 0.1, 'relative_time': 0.1, 'dwell_time': 0.1},
            {'type': 'press', 'key': 'h', 'timestamp': 0.2, 'relative_time': 0.2},
            {'type': 'release', 'key': 'h', 'timestamp': 0.25, 'relative_time': 0.25, 'dwell_time': 0.05, 'flight_time': 0.1}
        ]
        
        features = self.extractor._extract_statistical_features(session_data)
        
        self.assertIn('dwell_cv', features)
        self.assertIn('flight_cv', features)
    
    def test_extract_pattern_features(self):
        """Test pattern feature extraction."""
        session_data = [
            {'type': 'press', 'key': 't', 'timestamp': 0.0, 'relative_time': 0.0},
            {'type': 'press', 'key': 'h', 'timestamp': 0.1, 'relative_time': 0.1},
            {'type': 'press', 'key': 'e', 'timestamp': 0.2, 'relative_time': 0.2}
        ]
        
        features = self.extractor._extract_pattern_features(session_data)
        
        self.assertIn('avg_press_interval', features)
        self.assertIn('rhythm_consistency', features)
        self.assertIn('backspace_ratio', features)
    
    def test_extract_features_batch(self):
        """Test batch feature extraction."""
        sessions_data = [
            [
                {'type': 'press', 'key': 't', 'timestamp': 0.0, 'relative_time': 0.0},
                {'type': 'release', 'key': 't', 'timestamp': 0.1, 'relative_time': 0.1, 'dwell_time': 0.1}
            ],
            [
                {'type': 'press', 'key': 'h', 'timestamp': 0.0, 'relative_time': 0.0},
                {'type': 'release', 'key': 'h', 'timestamp': 0.1, 'relative_time': 0.1, 'dwell_time': 0.1}
            ]
        ]
        
        features_list = self.extractor.extract_features_batch(sessions_data)
        
        self.assertEqual(len(features_list), 2)
        self.assertGreater(len(features_list[0]), 0)
        self.assertGreater(len(features_list[1]), 0)
    
    def test_calculate_skewness(self):
        """Test skewness calculation."""
        data = [1, 2, 3, 4, 5]
        skewness = self.extractor._calculate_skewness(data)
        self.assertIsInstance(skewness, float)
        
        # Test with insufficient data
        skewness = self.extractor._calculate_skewness([1, 2])
        self.assertEqual(skewness, 0.0)
    
    def test_calculate_kurtosis(self):
        """Test kurtosis calculation."""
        data = [1, 2, 3, 4, 5]
        kurtosis = self.extractor._calculate_kurtosis(data)
        self.assertIsInstance(kurtosis, float)
        
        # Test with insufficient data
        kurtosis = self.extractor._calculate_kurtosis([1, 2, 3])
        self.assertEqual(kurtosis, 0.0)


class TestDataValidation(unittest.TestCase):
    """Test cases for data validation."""
    
    def test_session_data_validation(self):
        """Test session data validation."""
        # Valid session data
        valid_session = [
            {'type': 'press', 'key': 't', 'timestamp': 0.0, 'relative_time': 0.0},
            {'type': 'release', 'key': 't', 'timestamp': 0.1, 'relative_time': 0.1, 'dwell_time': 0.1}
        ]
        
        # Invalid session data (missing required fields)
        invalid_session = [
            {'type': 'press', 'key': 't'},  # Missing timestamp
            {'type': 'release', 'timestamp': 0.1}  # Missing key
        ]
        
        # Test that valid session doesn't raise errors
        try:
            extractor = FeatureExtractor()
            features = extractor.extract_features(valid_session)
            self.assertIsInstance(features, dict)
        except Exception as e:
            self.fail(f"Valid session data raised an exception: {e}")
        
        # Test that invalid session is handled gracefully
        try:
            features = extractor.extract_features(invalid_session)
            self.assertIsInstance(features, dict)
            # Should return empty features for invalid data
            self.assertEqual(features, {})
        except Exception as e:
            self.fail(f"Invalid session data should be handled gracefully: {e}")
    
    def test_feature_extraction_errors(self):
        """Test feature extraction error handling."""
        extractor = FeatureExtractor()
        
        # Test with None data
        features = extractor.extract_features(None)
        self.assertEqual(features, {})
        
        # Test with malformed data
        malformed_data = [
            {'type': 'press', 'key': None, 'timestamp': 'invalid'},
            {'type': 'release', 'key': 't', 'timestamp': 0.1, 'dwell_time': 'invalid'}
        ]
        
        try:
            features = extractor.extract_features(malformed_data)
            self.assertIsInstance(features, dict)
            # Should return empty features for malformed data
            self.assertEqual(features, {})
        except Exception as e:
            self.fail(f"Feature extraction should handle malformed data gracefully: {e}")


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2) 