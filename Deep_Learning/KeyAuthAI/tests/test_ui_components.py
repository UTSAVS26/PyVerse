"""
Test Cases for KeyAuthAI UI Components

This module contains comprehensive tests for:
- Authentication terminal functionality
- User verification
- Interactive mode
- Error handling in UI components
"""

import unittest
import tempfile
import os
import sys
from unittest.mock import patch, MagicMock
from typing import List, Dict

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ui.auth_terminal import AuthTerminal
from model.verify_user import UserVerifier


class TestAuthTerminal(unittest.TestCase):
    """Test cases for authentication terminal functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_file = os.path.join(self.temp_dir, "test_user_data.json")
        self.terminal = AuthTerminal()
        self.terminal.logger.data_file = self.data_file
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.data_file):
            os.remove(self.data_file)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)
    
    def test_initialization(self):
        """Test terminal initialization."""
        self.assertIsNotNone(self.terminal)
        self.assertIsNotNone(self.terminal.logger)
        self.assertIsNotNone(self.terminal.trainer)
        self.assertIsNotNone(self.terminal.verifier)
        self.assertEqual(self.terminal.default_passphrase, "the quick brown fox jumps over the lazy dog")
    
    @patch('ui.auth_terminal.input')
    @patch('ui.auth_terminal.KeystrokeLogger')
    def test_register_user_success(self, mock_logger_class, mock_input):
        """Test successful user registration."""
        # Mock logger
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger
        mock_logger.list_users.return_value = []
        mock_logger.start_recording.return_value = None
        mock_logger.stop_recording.return_value = [
            {'type': 'press', 'key': 't', 'timestamp': 0.0},
            {'type': 'release', 'key': 't', 'timestamp': 0.1, 'dwell_time': 0.1},
            {'type': 'press', 'key': 'h', 'timestamp': 0.2},
            {'type': 'release', 'key': 'h', 'timestamp': 0.25, 'dwell_time': 0.05, 'flight_time': 0.1},
            {'type': 'press', 'key': 'e', 'timestamp': 0.3},
            {'type': 'release', 'key': 'e', 'timestamp': 0.35, 'dwell_time': 0.05, 'flight_time': 0.05}
        ]
        
        # Mock trainer
        mock_trainer = MagicMock()
        mock_trainer.train_model.return_value = {
            'model_type': 'svm',
            'username': 'testuser',
            'accuracy': 0.95,
            'n_sessions': 5,
            'n_features': 10
        }
        mock_trainer.save_model.return_value = "model/model_testuser_svm.pkl"
        self.terminal.trainer = mock_trainer
        
        # Mock input to simulate user pressing Enter
        mock_input.return_value = ""
        
        # Test registration
        result = self.terminal.register_user('testuser', sessions=3)
        
        # Check results
        self.assertTrue(result['success'])
        self.assertEqual(result['username'], 'testuser')
        self.assertEqual(result['sessions_collected'], 3)
        self.assertIn('model_path', result)
        self.assertIn('training_results', result)
    
    @patch('ui.auth_terminal.input')
    def test_register_user_existing(self, mock_input):
        """Test registration with existing user."""
        # Mock existing user
        self.terminal.logger.user_data = {"testuser": {"sessions": []}}
        
        # Mock input to simulate user choosing not to overwrite
        mock_input.return_value = "n"
        
        # Test registration
        result = self.terminal.register_user('testuser')
        
        # Check results
        self.assertFalse(result['success'])
        self.assertEqual(result['error'], 'Registration cancelled by user')
    
    @patch('ui.auth_terminal.input')
    @patch('ui.auth_terminal.KeystrokeLogger')
    def test_register_user_insufficient_sessions(self, mock_logger_class, mock_input):
        """Test registration with insufficient sessions."""
        # Mock logger
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger
        mock_logger.list_users.return_value = []
        mock_logger.start_recording.return_value = None
        mock_logger.stop_recording.return_value = []  # Empty session
        
        # Mock input
        mock_input.return_value = ""
        
        # Test registration
        result = self.terminal.register_user('testuser', sessions=3)
        
        # Check results
        self.assertFalse(result['success'])
        self.assertEqual(result['error'], 'Insufficient training data')
    
    @patch('ui.auth_terminal.input')
    def test_authenticate_user_success(self, mock_input):
        """Test successful user authentication."""
        # Mock user exists
        self.terminal.logger.user_data = {
            "testuser": {
                "sessions": [],
                "passphrase": "test passphrase"
            }
        }
        
        # Mock verifier
        mock_verifier = MagicMock()
        mock_verifier.get_user_stats.return_value = {
            'username': 'testuser',
            'n_sessions': 5,
            'available_models': ['svm'],
            'passphrase': 'test passphrase'
        }
        mock_verifier.verify_user_interactive.return_value = {
            'authenticated': True,
            'confidence': 0.85,
            'threshold': 0.5,
            'model_type': 'svm',
            'session_length': 10,
            'error': None
        }
        self.terminal.verifier = mock_verifier
        
        # Mock input
        mock_input.return_value = ""
        
        # Test authentication
        result = self.terminal.authenticate_user('testuser')
        
        # Check results
        self.assertTrue(result['success'])
        self.assertTrue(result['authenticated'])
        self.assertEqual(result['confidence'], 0.85)
        self.assertEqual(result['model_type'], 'svm')
    
    def test_authenticate_user_not_found(self):
        """Test authentication with non-existent user."""
        # Mock empty user list
        self.terminal.logger.user_data = {}
        
        # Test authentication
        result = self.terminal.authenticate_user('nonexistent')
        
        # Check results
        self.assertFalse(result['success'])
        self.assertEqual(result['error'], 'User not found')
    
    def test_list_users_empty(self):
        """Test listing users when none exist."""
        # Mock empty user list
        self.terminal.logger.user_data = {}
        
        users = self.terminal.list_users()
        
        self.assertEqual(users, [])
    
    def test_list_users_with_data(self):
        """Test listing users when data exists."""
        # Mock user data
        self.terminal.logger.user_data = {
            "user1": {"sessions": []},
            "user2": {"sessions": []}
        }
        
        # Mock verifier stats
        mock_verifier = MagicMock()
        mock_verifier.get_user_stats.return_value = {
            'username': 'user1',
            'n_sessions': 5,
            'available_models': ['svm'],
            'error': None
        }
        self.terminal.verifier = mock_verifier
        
        users = self.terminal.list_users()
        
        self.assertEqual(len(users), 2)
        self.assertIn('user1', users)
        self.assertIn('user2', users)
    
    @patch('ui.auth_terminal.input')
    def test_delete_user_success(self, mock_input):
        """Test successful user deletion."""
        # Mock user exists
        self.terminal.logger.user_data = {
            "testuser": {"sessions": []},
            "otheruser": {"sessions": []}
        }
        
        # Mock input to confirm deletion
        mock_input.return_value = "DELETE"
        
        # Test deletion
        result = self.terminal.delete_user('testuser')
        
        # Check results
        self.assertTrue(result)
        self.assertNotIn('testuser', self.terminal.logger.user_data)
        self.assertIn('otheruser', self.terminal.logger.user_data)
    
    @patch('ui.auth_terminal.input')
    def test_delete_user_cancelled(self, mock_input):
        """Test cancelled user deletion."""
        # Mock user exists
        self.terminal.logger.user_data = {
            "testuser": {"sessions": []}
        }
        
        # Mock input to cancel deletion
        mock_input.return_value = "cancel"
        
        # Test deletion
        result = self.terminal.delete_user('testuser')
        
        # Check results
        self.assertFalse(result)
        self.assertIn('testuser', self.terminal.logger.user_data)
    
    def test_delete_user_not_found(self):
        """Test deletion of non-existent user."""
        # Mock empty user list
        self.terminal.logger.user_data = {}
        
        # Test deletion
        result = self.terminal.delete_user('nonexistent')
        
        # Check results
        self.assertFalse(result)


class TestUserVerifier(unittest.TestCase):
    """Test cases for user verification functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_file = os.path.join(self.temp_dir, "test_user_data.json")
        self.verifier = UserVerifier(self.data_file)
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.data_file):
            os.remove(self.data_file)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)
    
    def test_initialization(self):
        """Test verifier initialization."""
        self.assertIsNotNone(self.verifier)
        self.assertEqual(self.verifier.data_file, self.data_file)
        self.assertIsNotNone(self.verifier.trainer)
        self.assertIsNotNone(self.verifier.logger)
        self.assertIn('svm', self.verifier.thresholds)
        self.assertIn('one_class_svm', self.verifier.thresholds)
    
    def test_verify_user_not_found(self):
        """Test verification of non-existent user."""
        # Mock empty user list
        self.verifier.logger.user_data = {}
        
        session_data = [
            {'type': 'press', 'key': 't', 'timestamp': 0.0},
            {'type': 'release', 'key': 't', 'timestamp': 0.1, 'dwell_time': 0.1}
        ]
        
        result = self.verifier.verify_user('nonexistent', session_data)
        
        self.assertFalse(result['authenticated'])
        self.assertEqual(result['confidence'], 0.0)
        self.assertEqual(result['error'], 'User not found')
    
    @patch('os.path.exists')
    def test_verify_user_no_model(self, mock_exists):
        """Test verification when no model exists."""
        # Mock user exists but no model file
        self.verifier.logger.user_data = {"testuser": {"sessions": []}}
        mock_exists.return_value = False
        
        session_data = [
            {'type': 'press', 'key': 't', 'timestamp': 0.0},
            {'type': 'release', 'key': 't', 'timestamp': 0.1, 'dwell_time': 0.1}
        ]
        
        result = self.verifier.verify_user('testuser', session_data)
        
        self.assertFalse(result['authenticated'])
        self.assertEqual(result['confidence'], 0.0)
        self.assertEqual(result['error'], 'No trained model found for user')
    
    @patch('os.path.exists')
    @patch('model.verify_user.KeystrokeModelTrainer')
    def test_verify_user_success(self, mock_trainer_class, mock_exists):
        """Test successful user verification."""
        # Mock user exists and model file exists
        self.verifier.logger.user_data = {"testuser": {"sessions": []}}
        mock_exists.return_value = True
        
        # Mock trainer
        mock_trainer = MagicMock()
        mock_trainer_class.return_value = mock_trainer
        mock_trainer.load_model.return_value = {
            'model_type': 'svm',
            'username': 'testuser',
            'scaler': MagicMock(),
            'feature_names': ['feature1', 'feature2']
        }
        mock_trainer.predict.return_value = (0.85, {'feature1': 0.1, 'feature2': 0.2})
        self.verifier.trainer = mock_trainer
        
        session_data = [
            {'type': 'press', 'key': 't', 'timestamp': 0.0},
            {'type': 'release', 'key': 't', 'timestamp': 0.1, 'dwell_time': 0.1}
        ]
        
        result = self.verifier.verify_user('testuser', session_data)
        
        self.assertTrue(result['authenticated'])
        self.assertEqual(result['confidence'], 0.85)
        self.assertEqual(result['threshold'], 0.5)
        self.assertEqual(result['model_type'], 'svm')
        self.assertIsNone(result['error'])
    
    def test_get_user_stats(self):
        """Test getting user statistics."""
        # Mock user data
        self.verifier.logger.user_data = {
            "testuser": {
                "sessions": [{'timestamp': 1234567890, 'data': []}],
                "passphrase": "test passphrase"
            }
        }
        
        # Mock available models
        with patch.object(self.verifier, 'list_available_models') as mock_list:
            mock_list.return_value = ['svm', 'random_forest']
            
            stats = self.verifier.get_user_stats('testuser')
            
            self.assertEqual(stats['username'], 'testuser')
            self.assertEqual(stats['n_sessions'], 1)
            self.assertEqual(stats['passphrase'], 'test passphrase')
            self.assertEqual(stats['available_models'], ['svm', 'random_forest'])
    
    def test_get_user_stats_not_found(self):
        """Test getting stats for non-existent user."""
        # Mock empty user list
        self.verifier.logger.user_data = {}
        
        stats = self.verifier.get_user_stats('nonexistent')
        
        self.assertIn('error', stats)
        self.assertEqual(stats['error'], 'User not found')
    
    def test_list_available_models(self):
        """Test listing available models."""
        # Mock model files
        with patch('os.path.exists') as mock_exists:
            mock_exists.side_effect = lambda path: 'svm' in path or 'random_forest' in path
            
            models = self.verifier.list_available_models('testuser')
            
            self.assertIn('svm', models)
            self.assertIn('random_forest', models)
            self.assertNotIn('knn', models)
    
    def test_set_and_get_threshold(self):
        """Test setting and getting thresholds."""
        # Test setting threshold
        self.verifier.set_threshold('svm', 0.7)
        self.assertEqual(self.verifier.get_threshold('svm'), 0.7)
        
        # Test getting default threshold
        self.assertEqual(self.verifier.get_threshold('nonexistent'), 0.5)
    
    def test_batch_verify(self):
        """Test batch verification."""
        # Mock verification results
        with patch.object(self.verifier, 'verify_user') as mock_verify:
            mock_verify.side_effect = [
                {'authenticated': True, 'confidence': 0.8},
                {'authenticated': False, 'confidence': 0.3},
                {'authenticated': True, 'confidence': 0.9}
            ]
            
            test_sessions = [
                [{'type': 'press', 'key': 't'}],
                [{'type': 'press', 'key': 'h'}],
                [{'type': 'press', 'key': 'e'}]
            ]
            
            result = self.verifier.batch_verify('testuser', test_sessions)
            
            self.assertEqual(result['username'], 'testuser')
            self.assertEqual(result['total_sessions'], 3)
            self.assertEqual(result['authenticated_sessions'], 2)
            self.assertEqual(result['success_rate'], 2/3)
            self.assertEqual(len(result['session_results']), 3)


class TestErrorHandling(unittest.TestCase):
    """Test cases for error handling in UI components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_file = os.path.join(self.temp_dir, "test_user_data.json")
        self.terminal = AuthTerminal()
        self.terminal.logger.data_file = self.data_file
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.data_file):
            os.remove(self.data_file)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)
    
    def test_registration_error_handling(self):
        """Test error handling during registration."""
        # Mock logger to raise exception
        with patch.object(self.terminal.logger, 'start_recording') as mock_start:
            mock_start.side_effect = Exception("Recording error")
            
            result = self.terminal.register_user('testuser', sessions=1)
            
            self.assertFalse(result['success'])
            self.assertIn('Insufficient training data', result['error'])
    
    def test_authentication_error_handling(self):
        """Test error handling during authentication."""
        # Mock verifier to raise exception
        with patch.object(self.terminal.verifier, 'verify_user_interactive') as mock_verify:
            mock_verify.side_effect = Exception("Verification error")
            
            result = self.terminal.authenticate_user('testuser')
            
            self.assertFalse(result['success'])
            self.assertIn('User not found', result['error'])
    
    def test_file_io_error_handling(self):
        """Test file I/O error handling."""
        # Test with invalid file path
        terminal = AuthTerminal()
        terminal.logger.data_file = "/invalid/path/test.json"
        
        # Should not raise exception during initialization
        self.assertIsNotNone(terminal)
        
        # Test that operations handle file errors gracefully
        users = terminal.list_users()
        self.assertEqual(users, [])


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2) 