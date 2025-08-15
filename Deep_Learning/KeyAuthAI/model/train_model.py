"""
Model Training Module for KeyAuthAI

This module trains machine learning models for keystroke dynamics authentication:
- Supervised models: SVM, Random Forest, KNN
- Unsupervised models: One-Class SVM, Isolation Forest
- Model evaluation and metrics
"""

import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.svm import SVC, OneClassSVM
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.keystroke_logger import KeystrokeLogger
from features.extractor import FeatureExtractor


class KeystrokeModelTrainer:
    """Trains and manages keystroke dynamics authentication models."""
    
    def __init__(self, data_file: str = "data/user_data.json"):
        """
        Initialize the model trainer.
        
        Args:
            data_file: Path to user data file
        """
        self.data_file = data_file
        self.logger = KeystrokeLogger(data_file)
        self.extractor = FeatureExtractor()
        self.scaler = StandardScaler()
        self.model = None
        self.model_type = None
        self.feature_names = []
        
        # Available model types
        self.supervised_models = {
            'svm': SVC(kernel='rbf', probability=True),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'knn': KNeighborsClassifier(n_neighbors=3)
        }
        
        self.unsupervised_models = {
            'one_class_svm': OneClassSVM(kernel='rbf', nu=0.1),
            'isolation_forest': IsolationForest(contamination=0.1, random_state=42)
        }
    
    def train_model(self, username: str, model_type: str = 'svm', 
                   min_sessions: int = 5) -> Dict[str, Any]:
        """
        Train a model for a specific user.
        
        Args:
            username: Name of the user
            model_type: Type of model to train ('svm', 'random_forest', 'knn', 'one_class_svm', 'isolation_forest')
            min_sessions: Minimum number of sessions required for training
            
        Returns:
            Dictionary with training results and metrics
        """
        # Get user sessions
        sessions = self.logger.get_user_sessions(username)
        
        if len(sessions) < min_sessions:
            raise ValueError(f"Not enough sessions for user {username}. "
                          f"Required: {min_sessions}, Available: {len(sessions)}")
        
        # Extract features from all sessions
        session_data_list = [session['data'] for session in sessions]
        features_list = self.extractor.extract_features_batch(session_data_list)
        
        # Convert to DataFrame
        df = pd.DataFrame(features_list)
        self.feature_names = list(df.columns)
        
        # Handle missing values
        df = df.fillna(0)
        
        # Prepare data
        X = df.values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model based on type
        if model_type in self.supervised_models:
            return self._train_supervised_model(X_scaled, model_type, username)
        elif model_type in self.unsupervised_models:
            return self._train_unsupervised_model(X_scaled, model_type, username)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _train_supervised_model(self, X: np.ndarray, model_type: str, 
                               username: str) -> Dict[str, Any]:
        """Train a supervised model."""
        # For supervised models, we treat all sessions as positive samples
        # In a real scenario, you might have negative samples from other users
        y = np.ones(len(X))  # All samples are from the legitimate user
        
        # For single-class scenarios, we need to create synthetic negative samples
        # or use a different approach. Here we'll create synthetic negative samples
        if len(np.unique(y)) == 1:
            # Create synthetic negative samples by adding noise to positive samples
            n_samples = len(X)
            n_synthetic = min(n_samples, 10)  # Create up to 10 synthetic samples
            
            # Generate synthetic negative samples by adding noise
            # Generate synthetic negative samples by adding noise
            # Use adaptive noise based on feature scale
            noise_factor = 0.3  # Increased base factor
            feature_std = np.std(X, axis=0)
            feature_std[feature_std == 0] = 1.0  # Avoid division by zero
            synthetic_X = X[:n_synthetic] + np.random.normal(0, noise_factor * feature_std, X[:n_synthetic].shape)
            synthetic_y = np.zeros(n_synthetic)
            
            # Combine real and synthetic data
            X_combined = np.vstack([X, synthetic_X])
            y_combined = np.concatenate([y, synthetic_y])
        else:
            X_combined = X
            y_combined = y
        
        # Split data for evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, y_combined, test_size=0.2, random_state=42, stratify=y_combined
        )
        
        # Get model
        model = self.supervised_models[model_type]
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_combined, y_combined, cv=min(5, len(X_combined)))
        
        # Store model and metadata
        self.model = model
        self.model_type = model_type
        
        results = {
            'model_type': model_type,
            'username': username,
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'n_sessions': len(X),
            'n_features': X.shape[1],
            'feature_names': self.feature_names,
            'n_synthetic_samples': n_synthetic if len(np.unique(y)) == 1 else 0
        }
        
        return results
    
    def _train_unsupervised_model(self, X: np.ndarray, model_type: str, 
                                 username: str) -> Dict[str, Any]:
        """Train an unsupervised model."""
        # Get model
        model = self.unsupervised_models[model_type]
        
        # Train model (unsupervised models don't need labels)
        model.fit(X)
        
        # Evaluate model (predict on training data)
        predictions = model.predict(X)
        
        # For anomaly detection, -1 means anomaly, 1 means normal
        # We expect most predictions to be 1 (normal) for the legitimate user
        normal_predictions = np.sum(predictions == 1)
        anomaly_predictions = np.sum(predictions == -1)
        
        # Calculate anomaly rate
        anomaly_rate = anomaly_predictions / len(predictions)
        
        # Store model and metadata
        self.model = model
        self.model_type = model_type
        
        results = {
            'model_type': model_type,
            'username': username,
            'anomaly_rate': anomaly_rate,
            'normal_predictions': normal_predictions,
            'anomaly_predictions': anomaly_predictions,
            'n_sessions': len(X),
            'n_features': X.shape[1],
            'feature_names': self.feature_names
        }
        
        return results
    
    def save_model(self, username: str, model_path: str = None) -> str:
        """
        Save the trained model to disk.
        
        Args:
            username: Name of the user
            model_path: Path to save the model (optional)
            
        Returns:
            Path where model was saved
        """
        if self.model is None:
            raise ValueError("No model has been trained yet.")
        
        if model_path is None:
            model_path = f"model/model_{username}_{self.model_type}.pkl"
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model and metadata
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'username': username
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to: {model_path}")
        return model_path
    
    def load_model(self, model_path: str) -> Dict[str, Any]:
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            Dictionary with model and metadata
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        
        return model_data
    
    def predict(self, session_data: List[Dict]) -> Tuple[float, Dict[str, float]]:
        """
        Make a prediction on new session data.
        
        Args:
            session_data: List of keystroke events
            
        Returns:
            Tuple of (prediction_score, features_dict)
        """
        if self.model is None:
            raise ValueError("No model loaded. Please train or load a model first.")
        
        # Extract features
        features = self.extractor.extract_features(session_data)
        
        # Convert to array
        X = np.array([list(features.values())])
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make prediction
        if self.model_type in self.supervised_models:
            # For supervised models, get probability
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(X_scaled)[0]
                prediction_score = proba[1]  # Probability of being legitimate user
            else:
                prediction = self.model.predict(X_scaled)[0]
                prediction_score = 1.0 if prediction == 1 else 0.0
        else:
            # For unsupervised models, get decision function
            if hasattr(self.model, 'decision_function'):
                prediction_score = self.model.decision_function(X_scaled)[0]
            else:
                prediction = self.model.predict(X_scaled)[0]
                prediction_score = 1.0 if prediction == 1 else -1.0
        
        return prediction_score, features
    
    def evaluate_model(self, username: str, test_sessions: List[List[Dict]] = None) -> Dict[str, Any]:
        """
        Evaluate the trained model.
        
        Args:
            username: Name of the user
            test_sessions: Optional test sessions (if not provided, uses all sessions)
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("No model loaded. Please train or load a model first.")
        
        if test_sessions is None:
            # Use all sessions for evaluation
            sessions = self.logger.get_user_sessions(username)
            test_sessions = [session['data'] for session in sessions]
        
        predictions = []
        features_list = []
        
        for session_data in test_sessions:
            score, features = self.predict(session_data)
            predictions.append(score)
            features_list.append(features)
        
        # Calculate metrics
        if self.model_type in self.supervised_models:
            # For supervised models, higher score = more likely legitimate
            avg_score = np.mean(predictions)
            std_score = np.std(predictions)
            min_score = np.min(predictions)
            max_score = np.max(predictions)
        else:
            # For unsupervised models, higher score = more likely legitimate
            avg_score = np.mean(predictions)
            std_score = np.std(predictions)
            min_score = np.min(predictions)
            max_score = np.max(predictions)
        
        results = {
            'username': username,
            'model_type': self.model_type,
            'n_test_sessions': len(test_sessions),
            'avg_prediction_score': avg_score,
            'std_prediction_score': std_score,
            'min_prediction_score': min_score,
            'max_prediction_score': max_score,
            'predictions': predictions
        }
        
        return results


def train_user_model(username: str, model_type: str = 'svm', 
                    min_sessions: int = 5) -> Dict[str, Any]:
    """
    Convenience function to train a model for a user.
    
    Args:
        username: Name of the user
        model_type: Type of model to train
        min_sessions: Minimum number of sessions required
        
    Returns:
        Dictionary with training results
    """
    trainer = KeystrokeModelTrainer()
    results = trainer.train_model(username, model_type, min_sessions)
    
    # Save model
    model_path = trainer.save_model(username)
    results['model_path'] = model_path
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train keystroke dynamics model')
    parser.add_argument('--username', required=True, help='Username to train model for')
    parser.add_argument('--model', default='svm', 
                       choices=['svm', 'random_forest', 'knn', 'one_class_svm', 'isolation_forest'],
                       help='Model type to train')
    parser.add_argument('--min_sessions', type=int, default=5, 
                       help='Minimum number of sessions required')
    
    args = parser.parse_args()
    
    try:
        results = train_user_model(args.username, args.model, args.min_sessions)
        
        print(f"\nTraining Results for {args.username}:")
        print("=" * 40)
        for key, value in results.items():
            if key != 'feature_names':
                print(f"{key}: {value}")
        
        print(f"\nModel saved to: {results['model_path']}")
        
    except Exception as e:
        print(f"Error training model: {e}") 