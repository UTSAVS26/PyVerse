"""
Test Cases for KeyAuthAI Model Training

This module contains comprehensive tests for:
- Model training functionality
- Feature extraction and preprocessing
- Model evaluation
- Model saving and loading
"""

import unittest
import tempfile
import os
import sys
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from typing import List, Dict

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.train_model import KeystrokeModelTrainer
from features.extractor import FeatureExtractor


class TestModelTrainer(unittest.TestCase):
    """Test cases for model training functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_file = os.path.join(self.temp_dir, "test_user_data.json")
        self.trainer = KeystrokeModelTrainer(self.data_file)
        
        # Create sample session data
        self.sample_sessions = [
            [
                {'type': 'press', 'key': 't', 'timestamp': 0.0, 'relative_time': 0.0},
                {'type': 'release', 'key': 't', 'timestamp': 0.1, 'relative_time': 0.1, 'dwell_time': 0.1},
                {'type': 'press', 'key': 'h', 'timestamp': 0.2, 'relative_time': 0.2},
                {'type': 'release', 'key': 'h', 'timestamp': 0.25, 'relative_time': 0.25, 'dwell_time': 0.05, 'flight_time': 0.1}
            ],
            [
                {'type': 'press', 'key': 't', 'timestamp': 0.0, 'relative_time': 0.0},
                {'type': 'release', 'key': 't', 'timestamp': 0.12, 'relative_time': 0.12, 'dwell_time': 0.12},
                {'type': 'press', 'key': 'h', 'timestamp': 0.22, 'relative_time': 0.22},
                {'type': 'release', 'key': 'h', 'timestamp': 0.27, 'relative_time': 0.27, 'dwell_time': 0.05, 'flight_time': 0.1}
            ],
            [
                {'type': 'press', 'key': 't', 'timestamp': 0.0, 'relative_time': 0.0},
                {'type': 'release', 'key': 't', 'timestamp': 0.11, 'relative_time': 0.11, 'dwell_time': 0.11},
                {'type': 'press', 'key': 'h', 'timestamp': 0.21, 'relative_time': 0.21},
                {'type': 'release', 'key': 'h', 'timestamp': 0.26, 'relative_time': 0.26, 'dwell_time': 0.05, 'flight_time': 0.1}
            ]
        ]
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up data file
        if os.path.exists(self.data_file):
            os.remove(self.data_file)
        
        # Clean up model files
        for model_type in ['svm', 'random_forest', 'knn', 'one_class_svm', 'isolation_forest']:
            model_path = f"model/model_testuser_{model_type}.pkl"
            if os.path.exists(model_path):
                os.remove(model_path)
        
        # Clean up directories safely
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        # Don't remove the model directory as it may be used by other tests
    def test_initialization(self):
        """Test trainer initialization."""
        self.assertIsNotNone(self.trainer)
        self.assertEqual(self.trainer.data_file, self.data_file)
        self.assertIsNone(self.trainer.model)
        self.assertIsNone(self.trainer.model_type)
        self.assertEqual(self.trainer.feature_names, [])
    
    def test_supervised_models_available(self):
        """Test that supervised models are available."""
        self.assertIn('svm', self.trainer.supervised_models)
        self.assertIn('random_forest', self.trainer.supervised_models)
        self.assertIn('knn', self.trainer.supervised_models)
    
    def test_unsupervised_models_available(self):
        """Test that unsupervised models are available."""
        self.assertIn('one_class_svm', self.trainer.unsupervised_models)
        self.assertIn('isolation_forest', self.trainer.unsupervised_models)
    
    @patch('model.train_model.KeystrokeLogger')
    def test_train_supervised_model(self, mock_logger):
        """Test supervised model training."""
        # Mock logger to return sample sessions
        mock_logger_instance = MagicMock()
        mock_logger_instance.get_user_sessions.return_value = [
            {'data': session} for session in self.sample_sessions
        ]
        mock_logger.return_value = mock_logger_instance
        
        # Create trainer with mocked logger
        trainer = KeystrokeModelTrainer(self.data_file)
        trainer.logger = mock_logger_instance
        
        # Train SVM model
        results = trainer.train_model('testuser', 'svm', min_sessions=3)
        
        # Check results
        self.assertIsNotNone(results)
        self.assertEqual(results['model_type'], 'svm')
        self.assertEqual(results['username'], 'testuser')
        self.assertIn('n_sessions', results)
        self.assertIn('n_features', results)
        self.assertIn('accuracy', results)
        self.assertIn('cv_mean', results)
        
        # Check that model was trained
        self.assertIsNotNone(trainer.model)
        self.assertEqual(trainer.model_type, 'svm')
        self.assertGreater(len(trainer.feature_names), 0)
    
    @patch('model.train_model.KeystrokeLogger')
    def test_train_unsupervised_model(self, mock_logger):
        """Test unsupervised model training."""
        # Mock logger to return sample sessions
        mock_logger_instance = MagicMock()
        mock_logger_instance.get_user_sessions.return_value = [
            {'data': session} for session in self.sample_sessions
        ]
        mock_logger.return_value = mock_logger_instance
        
        # Create trainer with mocked logger
        trainer = KeystrokeModelTrainer(self.data_file)
        trainer.logger = mock_logger_instance
        
        # Train One-Class SVM model
        results = trainer.train_model('testuser', 'one_class_svm', min_sessions=3)
        
        # Check results
        self.assertIsNotNone(results)
        self.assertEqual(results['model_type'], 'one_class_svm')
        self.assertEqual(results['username'], 'testuser')
        self.assertIn('n_sessions', results)
        self.assertIn('n_features', results)
        self.assertIn('anomaly_rate', results)
        
        # Check that model was trained
        self.assertIsNotNone(trainer.model)
        self.assertEqual(trainer.model_type, 'one_class_svm')
        self.assertGreater(len(trainer.feature_names), 0)
    
    def test_train_model_insufficient_sessions(self):
        """Test training with insufficient sessions."""
        with self.assertRaises(ValueError):
            self.trainer.train_model('testuser', 'svm', min_sessions=10)
    
    def test_train_model_invalid_type(self):
        """Test training with invalid model type."""
        with self.assertRaises(ValueError):
            self.trainer.train_model('testuser', 'invalid_model', min_sessions=3)
    
    def test_save_and_load_model(self):
        """Test model saving and loading."""
        # Create sample features and model
        self.trainer.feature_names = ['feature1', 'feature2']
        self.trainer.model_type = 'svm'
        
        # Create a real model instead of MagicMock for pickling
        from sklearn.svm import SVC
        from sklearn.preprocessing import StandardScaler
        self.trainer.model = SVC()
        self.trainer.scaler = StandardScaler()
        # Fit the scaler with dummy data to make it valid for pickling
        self.trainer.scaler.fit([[1, 2], [3, 4]])
        # Save model
        model_path = self.trainer.save_model('testuser')
        
        # Verify file was created
        self.assertTrue(os.path.exists(model_path))
        
        # Load model
        loaded_data = self.trainer.load_model(model_path)
        
        # Check loaded data
        self.assertIn('model', loaded_data)
        self.assertIn('model_type', loaded_data)
        self.assertIn('scaler', loaded_data)
        self.assertIn('feature_names', loaded_data)
        self.assertIn('username', loaded_data)
        
        self.assertEqual(loaded_data['model_type'], 'svm')
        self.assertEqual(loaded_data['username'], 'testuser')
        self.assertEqual(loaded_data['feature_names'], ['feature1', 'feature2'])
    
    def test_load_model_nonexistent(self):
        """Test loading non-existent model."""
        with self.assertRaises(FileNotFoundError):
            self.trainer.load_model('nonexistent_model.pkl')
    
    def test_predict_without_model(self):
        """Test prediction without trained model."""
        session_data = [
            {'type': 'press', 'key': 't', 'timestamp': 0.0, 'relative_time': 0.0},
            {'type': 'release', 'key': 't', 'timestamp': 0.1, 'relative_time': 0.1, 'dwell_time': 0.1}
        ]
        
        with self.assertRaises(ValueError):
            self.trainer.predict(session_data)
    
    @patch('model.train_model.KeystrokeLogger')
    def test_predict_with_model(self, mock_logger):
        """Test prediction with trained model."""
        # Mock logger
        mock_logger_instance = MagicMock()
        mock_logger_instance.get_user_sessions.return_value = [
            {'data': session} for session in self.sample_sessions
        ]
        mock_logger.return_value = mock_logger_instance
        
        # Create trainer and train model
        trainer = KeystrokeModelTrainer(self.data_file)
        trainer.logger = mock_logger_instance
        trainer.train_model('testuser', 'svm', min_sessions=3)
        
        # Test prediction
        session_data = [
            {'type': 'press', 'key': 't', 'timestamp': 0.0, 'relative_time': 0.0},
            {'type': 'release', 'key': 't', 'timestamp': 0.1, 'relative_time': 0.1, 'dwell_time': 0.1}
        ]
        
        score, features = trainer.predict(session_data)
        
        # Check prediction results
        self.assertIsInstance(score, float)
        self.assertIsInstance(features, dict)
        self.assertGreater(len(features), 0)
    
    @patch('model.train_model.KeystrokeLogger')
    def test_evaluate_model(self, mock_logger):
        """Test model evaluation."""
        # Mock logger
        mock_logger_instance = MagicMock()
        mock_logger_instance.get_user_sessions.return_value = [
            {'data': session} for session in self.sample_sessions
        ]
        mock_logger.return_value = mock_logger_instance
        
        # Create trainer and train model
        trainer = KeystrokeModelTrainer(self.data_file)
        trainer.logger = mock_logger_instance
        trainer.train_model('testuser', 'svm', min_sessions=3)
        
        # Evaluate model
        results = trainer.evaluate_model('testuser')
        
        # Check evaluation results
        self.assertIsNotNone(results)
        self.assertIn('username', results)
        self.assertIn('model_type', results)
        self.assertIn('n_test_sessions', results)
        self.assertIn('avg_prediction_score', results)
        self.assertIn('predictions', results)
        
        self.assertEqual(results['username'], 'testuser')
        self.assertEqual(results['model_type'], 'svm')
        self.assertEqual(results['n_test_sessions'], 3)


class TestModelTypes(unittest.TestCase):
    """Test cases for different model types."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_file = os.path.join(self.temp_dir, "test_user_data.json")
        self.trainer = KeystrokeModelTrainer(self.data_file)
        
        # Create sample features
        self.sample_features = [
            {'avg_dwell_time': 0.1, 'avg_flight_time': 0.2, 'total_time': 1.0},
            {'avg_dwell_time': 0.12, 'avg_flight_time': 0.22, 'total_time': 1.1},
            {'avg_dwell_time': 0.11, 'avg_flight_time': 0.21, 'total_time': 1.05}
        ]
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.data_file):
            os.remove(self.data_file)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)
    
    def test_svm_model(self):
        """Test SVM model training."""
        # Create DataFrame from sample features with more samples
        sample_features = [
            {'avg_dwell_time': 0.1, 'avg_flight_time': 0.2, 'total_time': 1.0},
            {'avg_dwell_time': 0.12, 'avg_flight_time': 0.22, 'total_time': 1.1},
            {'avg_dwell_time': 0.11, 'avg_flight_time': 0.21, 'total_time': 1.05},
            {'avg_dwell_time': 0.13, 'avg_flight_time': 0.23, 'total_time': 1.15},
            {'avg_dwell_time': 0.14, 'avg_flight_time': 0.24, 'total_time': 1.25}
        ]
        
        df = pd.DataFrame(sample_features)
        X = df.values
        X_scaled = self.trainer.scaler.fit_transform(X)
        
        # Train SVM model
        results = self.trainer._train_supervised_model(X_scaled, 'svm', 'testuser')
        
        self.assertEqual(results['model_type'], 'svm')
        self.assertIn('accuracy', results)
        self.assertIn('cv_mean', results)
    
    def test_random_forest_model(self):
        """Test Random Forest model training."""
        # Create DataFrame from sample features
        df = pd.DataFrame(self.sample_features)
        X = df.values
        X_scaled = self.trainer.scaler.fit_transform(X)
        
        # Train Random Forest model
        results = self.trainer._train_supervised_model(X_scaled, 'random_forest', 'testuser')
        
        self.assertEqual(results['model_type'], 'random_forest')
        self.assertIn('accuracy', results)
        self.assertIn('cv_mean', results)
    
    def test_knn_model(self):
        """Test KNN model training."""
        # Create DataFrame from sample features with more samples
        sample_features = [
            {'avg_dwell_time': 0.1, 'avg_flight_time': 0.2, 'total_time': 1.0},
            {'avg_dwell_time': 0.12, 'avg_flight_time': 0.22, 'total_time': 1.1},
            {'avg_dwell_time': 0.11, 'avg_flight_time': 0.21, 'total_time': 1.05},
            {'avg_dwell_time': 0.13, 'avg_flight_time': 0.23, 'total_time': 1.15},
            {'avg_dwell_time': 0.14, 'avg_flight_time': 0.24, 'total_time': 1.25}
        ]
        
        df = pd.DataFrame(sample_features)
        X = df.values
        X_scaled = self.trainer.scaler.fit_transform(X)
        
        # Train KNN model
        results = self.trainer._train_supervised_model(X_scaled, 'knn', 'testuser')
        
        self.assertEqual(results['model_type'], 'knn')
        self.assertIn('accuracy', results)
        self.assertIn('cv_mean', results)
    
    def test_one_class_svm_model(self):
        """Test One-Class SVM model training."""
        # Create DataFrame from sample features
        df = pd.DataFrame(self.sample_features)
        X = df.values
        X_scaled = self.trainer.scaler.fit_transform(X)
        
        # Train One-Class SVM model
        results = self.trainer._train_unsupervised_model(X_scaled, 'one_class_svm', 'testuser')
        
        self.assertEqual(results['model_type'], 'one_class_svm')
        self.assertIn('anomaly_rate', results)
        self.assertIn('normal_predictions', results)
        self.assertIn('anomaly_predictions', results)
    
    def test_isolation_forest_model(self):
        """Test Isolation Forest model training."""
        # Create DataFrame from sample features
        df = pd.DataFrame(self.sample_features)
        X = df.values
        X_scaled = self.trainer.scaler.fit_transform(X)
        
        # Train Isolation Forest model
        results = self.trainer._train_unsupervised_model(X_scaled, 'isolation_forest', 'testuser')
        
        self.assertEqual(results['model_type'], 'isolation_forest')
        self.assertIn('anomaly_rate', results)
        self.assertIn('normal_predictions', results)
        self.assertIn('anomaly_predictions', results)


class TestFeaturePreprocessing(unittest.TestCase):
    """Test cases for feature preprocessing."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.trainer = KeystrokeModelTrainer()
    
    def test_feature_scaling(self):
        """Test feature scaling."""
        # Create sample features with different scales
        features = [
            {'avg_dwell_time': 0.1, 'total_time': 10.0, 'typing_speed_cps': 5.0},
            {'avg_dwell_time': 0.2, 'total_time': 20.0, 'typing_speed_cps': 10.0},
            {'avg_dwell_time': 0.15, 'total_time': 15.0, 'typing_speed_cps': 7.5}
        ]
        
        # Convert to DataFrame
        df = pd.DataFrame(features)
        X = df.values
        
        # Scale features
        X_scaled = self.trainer.scaler.fit_transform(X)
        
        # Check that scaling was applied
        self.assertNotEqual(X_scaled.tolist(), X.tolist())
        
        # Check that scaled features have mean close to 0 and std close to 1
        self.assertAlmostEqual(np.mean(X_scaled, axis=0)[0], 0.0, places=10)
        self.assertAlmostEqual(np.std(X_scaled, axis=0)[0], 1.0, places=10)
    
    def test_missing_value_handling(self):
        """Test handling of missing values."""
        # Create features with missing values
        features = [
            {'avg_dwell_time': 0.1, 'total_time': 10.0, 'typing_speed_cps': np.nan},
            {'avg_dwell_time': np.nan, 'total_time': 20.0, 'typing_speed_cps': 10.0},
            {'avg_dwell_time': 0.15, 'total_time': 15.0, 'typing_speed_cps': 7.5}
        ]
        
        # Convert to DataFrame
        df = pd.DataFrame(features)
        
        # Fill missing values
        df = df.fillna(0)
        
        # Check that no NaN values remain
        self.assertFalse(df.isnull().any().any())
        
        # Check that missing values were filled with 0
        self.assertEqual(df.iloc[0]['typing_speed_cps'], 0.0)
        self.assertEqual(df.iloc[1]['avg_dwell_time'], 0.0)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2) 