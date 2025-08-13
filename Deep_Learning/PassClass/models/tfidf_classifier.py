"""
TF-IDF Password Classifier

This module implements a TF-IDF based classifier for password strength prediction
using scikit-learn.
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from typing import Dict, List, Tuple, Any
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from labeling.labeler import PasswordLabeler


class TFIDFPasswordClassifier:
    """TF-IDF based password strength classifier."""
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize the classifier.
        
        Args:
            model_type: 'random_forest', 'logistic_regression', or 'svm'
        """
        self.model_type = model_type
        self.labeler = PasswordLabeler()
        
        # Initialize vectorizer
        self.vectorizer = TfidfVectorizer(
            analyzer='char',  # Character-level analysis
            ngram_range=(1, 3),  # 1-3 character n-grams
            max_features=1000,
            min_df=2,
            max_df=0.95
        )
        
        # Initialize classifier based on type
        if model_type == 'random_forest':
            self.classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        elif model_type == 'logistic_regression':
            self.classifier = LogisticRegression(
                max_iter=1000,
                random_state=42,
                multi_class='multinomial'
            )
        elif model_type == 'svm':
            self.classifier = SVC(
                kernel='rbf',
                probability=True,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Create pipeline
        self.pipeline = Pipeline([
            ('vectorizer', self.vectorizer),
            ('classifier', self.classifier)
        ])
        
        self.is_trained = False
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for training.
        
        Args:
            df: DataFrame with 'password' and 'label' columns
            
        Returns:
            Tuple of (X, y) for training
        """
def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare data for training.
    
    Args:
        df: DataFrame with 'password' and 'label' columns
        
    Returns:
        Tuple of (X, y) for training
    """
    required_columns = ['password', 'label']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"DataFrame missing required columns: {missing_columns}")
    
    X = df['password'].values
    y = df['label'].values
    
    return X, y
    
    def train(self, df: pd.DataFrame, test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train the classifier.
        
        Args:
            df: DataFrame with 'password' and 'label' columns
            test_size: Fraction of data to use for testing
            
        Returns:
            Dictionary with training results
        """
        print(f"Training {self.model_type} classifier...")
        
        # Prepare data
        X, y = self.prepare_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Train pipeline
        self.pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.pipeline.predict(X_test)
        y_pred_proba = self.pipeline.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(self.pipeline, X_train, y_train, cv=5)
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        self.is_trained = True
        
        results = {
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': report,
            'confusion_matrix': cm,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'feature_names': self.vectorizer.get_feature_names_out() if hasattr(self.vectorizer, 'get_feature_names_out') else []
        }
        
        print(f"Training completed!")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Cross-validation score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return results
    
    def predict(self, password: str) -> str:
        """
        Predict password strength.
        
        Args:
            password: Password to classify
            
        Returns:
            Predicted label ('weak', 'medium', 'strong')
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.pipeline.predict([password])[0]
    
    def predict_proba(self, password: str) -> Dict[str, float]:
        """
        Get prediction probabilities.
        
        Args:
            password: Password to classify
            
        Returns:
            Dictionary with probability for each class
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        proba = self.pipeline.predict_proba([password])[0]
        classes = self.pipeline.classes_
        
        return dict(zip(classes, proba))
    
    def predict_with_confidence(self, password: str) -> Dict[str, Any]:
        """
        Predict password strength with confidence and analysis.
        
        Args:
            password: Password to classify
            
        Returns:
            Dictionary with prediction details
        """
        prediction = self.predict(password)
        proba = self.predict_proba(password)
        confidence = max(proba.values())
        
        # Get detailed analysis from labeler
        analysis = self.labeler.get_detailed_analysis(password)
        
        return {
            'password': password,
            'predicted_label': prediction,
            'confidence': confidence,
            'probabilities': proba,
            'actual_analysis': analysis,
            'labeler_label': analysis['label'],
            'prediction_match': prediction == analysis['label']
        }
    
    def evaluate_on_test_set(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the model on a test set.
        
        Args:
            X_test: Test passwords
            y_test: True labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        y_pred = self.pipeline.predict(X_test)
        y_pred_proba = self.pipeline.predict_proba(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    def save_model(self, filepath: str):
        """Save the trained model to disk."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.pipeline, f)
        
        print(f"Model saved to: {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model from disk."""
        with open(filepath, 'rb') as f:
            self.pipeline = pickle.load(f)
        
        self.is_trained = True
        print(f"Model loaded from: {filepath}")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance (for Random Forest only).
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        if self.model_type != 'random_forest':
            raise ValueError("Feature importance only available for Random Forest")
        
        feature_names = self.vectorizer.get_feature_names_out()
        importance = self.classifier.feature_importances_
        
        return dict(zip(feature_names, importance))


def create_and_train_classifier(df: pd.DataFrame, model_type: str = 'random_forest') -> TFIDFPasswordClassifier:
    """
    Create and train a classifier.
    
    Args:
        df: Training data
        model_type: Type of classifier to use
        
    Returns:
        Trained classifier
    """
    classifier = TFIDFPasswordClassifier(model_type=model_type)
    results = classifier.train(df)
    
    return classifier, results


if __name__ == "__main__":
    # Test the classifier
    print("Testing TF-IDF Password Classifier...")
    
    # Load or generate sample data
    try:
        df = pd.read_csv('data/password_dataset.csv')
        print(f"Loaded dataset with {len(df)} samples")
    except FileNotFoundError:
        print("Dataset not found. Please run data/generate_passwords.py first.")
        exit(1)
    
    # Train classifier
    classifier, results = create_and_train_classifier(df, model_type='random_forest')
    
    # Test predictions
    test_passwords = [
        "abc",           # weak
        "password123",   # weak
        "Hello2023!",    # medium
        "G7^s9L!zB1m",  # strong
        "qwerty",       # weak
        "MyPass@123",   # medium
        "tR#8$!XmPq@",  # strong
    ]
    
    print(f"\nTesting predictions:")
    print("=" * 50)
    
    for password in test_passwords:
        result = classifier.predict_with_confidence(password)
        print(f"\nPassword: {password}")
        print(f"Predicted: {result['predicted_label']} (confidence: {result['confidence']:.3f})")
        print(f"Actual: {result['labeler_label']}")
        print(f"Match: {'✅' if result['prediction_match'] else '❌'}")
    
    # Save model
    classifier.save_model('models/tfidf_password_classifier.pkl') 