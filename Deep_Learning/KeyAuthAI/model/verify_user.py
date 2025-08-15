"""
User Verification Module for KeyAuthAI

This module handles user authentication using trained keystroke dynamics models:
- Load trained models
- Verify user identity
- Calculate confidence scores
- Handle authentication thresholds
"""

import os
import sys
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.train_model import KeystrokeModelTrainer
from data.keystroke_logger import KeystrokeLogger


class UserVerifier:
    """Handles user authentication using keystroke dynamics."""
    
    def __init__(self, data_file: str = "data/user_data.json"):
        """
        Initialize the user verifier.
        
        Args:
            data_file: Path to user data file
        """
        self.data_file = data_file
        self.trainer = KeystrokeModelTrainer(data_file)
        self.logger = KeystrokeLogger(data_file)
        
        # Default thresholds for different model types
        self.thresholds = {
            'svm': 0.5,
            'random_forest': 0.5,
            'knn': 0.5,
            'one_class_svm': 0.0,
            'isolation_forest': 0.0
        }
    
    def verify_user(self, username: str, session_data: List[Dict], 
                   threshold: float = None) -> Dict[str, Any]:
        """
        Verify a user's identity using keystroke dynamics.
        
        Args:
            username: Name of the user to verify
            session_data: Keystroke session data
            threshold: Authentication threshold (optional)
            
        Returns:
            Dictionary with verification results
        """
        # Check if user exists
        if username not in self.logger.list_users():
            return {
                'authenticated': False,
                'confidence': 0.0,
                'error': 'User not found'
            }
        
        # Check if model exists
        model_path = f"model/model_{username}_svm.pkl"
        if not os.path.exists(model_path):
            # Try other model types
            for model_type in ['random_forest', 'knn', 'one_class_svm', 'isolation_forest']:
                model_path = f"model/model_{username}_{model_type}.pkl"
                if os.path.exists(model_path):
                    break
            else:
                return {
                    'authenticated': False,
                    'confidence': 0.0,
                    'error': 'No trained model found for user'
                }
        
        try:
            # Load model
            model_data = self.trainer.load_model(model_path)
            model_type = model_data['model_type']
            
            # Make prediction
            prediction_score, features = self.trainer.predict(session_data)
            
            # Determine threshold
            if threshold is None:
                threshold = self.thresholds.get(model_type, 0.5)
            
            # Determine authentication result
            if model_type in ['svm', 'random_forest', 'knn']:
                # Supervised models: higher score = more likely legitimate
                authenticated = prediction_score >= threshold
                confidence = prediction_score
            else:
                # Unsupervised models: higher score = more likely legitimate
                authenticated = prediction_score >= threshold
                confidence = prediction_score
            
            return {
                'authenticated': authenticated,
                'confidence': confidence,
                'threshold': threshold,
                'model_type': model_type,
                'features': features,
                'error': None
            }
            
        except Exception as e:
            return {
                'authenticated': False,
                'confidence': 0.0,
                'error': f'Verification error: {str(e)}'
            }
    
    def verify_user_interactive(self, username: str, passphrase: str = None) -> Dict[str, Any]:
        """
        Verify a user interactively by recording their keystrokes.
        
        Args:
            username: Name of the user to verify
            passphrase: Passphrase to type (optional)
            
        Returns:
            Dictionary with verification results
        """
        # Get user's passphrase
        if passphrase is None:
            passphrase = self.logger.get_user_passphrase(username)
            if passphrase is None:
                return {
                    'authenticated': False,
                    'confidence': 0.0,
                    'error': 'No passphrase found for user'
                }
        
        print(f"Verifying user: {username}")
        print(f"Please type: '{passphrase}'")
        print("Press Enter when done...")
        
        try:
            # Record keystroke session
            self.logger.start_recording(username, passphrase)
            input()
            session_data = self.logger.stop_recording()
            
            # Verify user
            result = self.verify_user(username, session_data)
            
            # Add session info
            result['session_length'] = len(session_data)
            result['passphrase'] = passphrase
            
            return result
            
        except KeyboardInterrupt:
            print("\nVerification cancelled.")
            return {
                'authenticated': False,
                'confidence': 0.0,
                'error': 'Verification cancelled by user'
            }
        except Exception as e:
            return {
                'authenticated': False,
                'confidence': 0.0,
                'error': f'Verification error: {str(e)}'
            }
    
    def set_threshold(self, model_type: str, threshold: float):
        """
        Set authentication threshold for a model type.
        
        Args:
            model_type: Type of model
            threshold: Authentication threshold
        """
        self.thresholds[model_type] = threshold
    
    def get_threshold(self, model_type: str) -> float:
        """
        Get authentication threshold for a model type.
        
        Args:
            model_type: Type of model
            
        Returns:
            Authentication threshold
        """
        return self.thresholds.get(model_type, 0.5)
    
    def list_available_models(self, username: str) -> List[str]:
        """
        List available trained models for a user.
        
        Args:
            username: Name of the user
            
        Returns:
            List of available model types
        """
        available_models = []
        model_types = ['svm', 'random_forest', 'knn', 'one_class_svm', 'isolation_forest']
        
        for model_type in model_types:
            model_path = f"model/model_{username}_{model_type}.pkl"
            if os.path.exists(model_path):
                available_models.append(model_type)
        
        return available_models
    
    def get_user_stats(self, username: str) -> Dict[str, Any]:
        """
        Get statistics for a user.
        
        Args:
            username: Name of the user
            
        Returns:
            Dictionary with user statistics
        """
        if username not in self.logger.list_users():
            return {'error': 'User not found'}
        
        sessions = self.logger.get_user_sessions(username)
        passphrase = self.logger.get_user_passphrase(username)
        available_models = self.list_available_models(username)
        
        return {
            'username': username,
            'n_sessions': len(sessions),
            'passphrase': passphrase,
            'available_models': available_models,
            'created_at': sessions[0]['timestamp'] if sessions else None
        }
    
    def batch_verify(self, username: str, test_sessions: List[List[Dict]], 
                    threshold: float = None) -> Dict[str, Any]:
        """
        Verify a user against multiple test sessions.
        
        Args:
            username: Name of the user
            test_sessions: List of test session data
            threshold: Authentication threshold (optional)
            
        Returns:
            Dictionary with batch verification results
        """
        results = []
        authenticated_count = 0
        
        for i, session_data in enumerate(test_sessions):
            result = self.verify_user(username, session_data, threshold)
            result['session_index'] = i
            results.append(result)
            
            if result['authenticated']:
                authenticated_count += 1
        
        # Calculate overall metrics
        total_sessions = len(test_sessions)
        success_rate = authenticated_count / total_sessions if total_sessions > 0 else 0
        avg_confidence = np.mean([r['confidence'] for r in results])
        
        return {
            'username': username,
            'total_sessions': total_sessions,
            'authenticated_sessions': authenticated_count,
            'success_rate': success_rate,
            'avg_confidence': avg_confidence,
            'session_results': results
        }


def verify_user_identity(username: str, session_data: List[Dict]) -> Dict[str, Any]:
    """
    Convenience function to verify a user's identity.
    
    Args:
        username: Name of the user
        session_data: Keystroke session data
        
    Returns:
        Dictionary with verification results
    """
    verifier = UserVerifier()
    return verifier.verify_user(username, session_data)


def verify_user_interactive(username: str, passphrase: str = None) -> Dict[str, Any]:
    """
    Convenience function to verify a user interactively.
    
    Args:
        username: Name of the user
        passphrase: Passphrase to type (optional)
        
    Returns:
        Dictionary with verification results
    """
    verifier = UserVerifier()
    return verifier.verify_user_interactive(username, passphrase)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Verify user identity')
    parser.add_argument('--username', required=True, help='Username to verify')
    parser.add_argument('--passphrase', help='Passphrase to type (optional)')
    parser.add_argument('--threshold', type=float, help='Authentication threshold')
    
    args = parser.parse_args()
    
    verifier = UserVerifier()
    
    # Get user stats
    stats = verifier.get_user_stats(args.username)
    if 'error' in stats:
        print(f"Error: {stats['error']}")
        exit(1)
    
    print(f"User: {stats['username']}")
    print(f"Sessions: {stats['n_sessions']}")
    print(f"Available models: {', '.join(stats['available_models'])}")
    print()
    
    # Verify user
    result = verifier.verify_user_interactive(args.username, args.passphrase)
    
    print("\nVerification Results:")
    print("=" * 30)
    print(f"Authenticated: {result['authenticated']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Model Type: {result.get('model_type', 'N/A')}")
    print(f"Session Length: {result.get('session_length', 'N/A')}")
    
    if result['error']:
        print(f"Error: {result['error']}")
    
    if result['authenticated']:
        print("\n✅ Authentication successful!")
    else:
        print("\n❌ Authentication failed!") 