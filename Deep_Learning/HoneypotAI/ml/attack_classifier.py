"""
Attack Classifier for HoneypotAI
Uses supervised learning to classify known attack types
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import joblib
import structlog

logger = structlog.get_logger(__name__)

class AttackClassifier:
    """Supervised learning classifier for attack type classification"""
    
    def __init__(self, method: str = "random_forest"):
        self.method = method
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.confidence_threshold = 0.9
        self.is_trained = False
        self.feature_names = []
        self.class_names = []
        
        self.logger = structlog.get_logger("ml.attack_classifier")
        
        # Initialize model based on method
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the classification model"""
        if self.method == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif self.method == "logistic_regression":
            self.model = LogisticRegression(
                random_state=42,
                max_iter=1000,
                multi_class='ovr'
            )
        elif self.method == "svm":
            self.model = SVC(
                kernel='rbf',
                probability=True,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown classification method: {self.method}")
    
    def train(self, features: pd.DataFrame, logs: List[Dict[str, Any]]) -> bool:
        """Train the attack classifier"""
        try:
            if features.empty or not logs:
                self.logger.warning("No features or logs provided for training")
                return False
            
            # Extract labels from logs
            labels = []
            for log in logs:
                attack_type = log.get('attack_type', 'none')
                labels.append(attack_type)
            
            # Encode labels
            y = self.label_encoder.fit_transform(labels)
            self.class_names = self.label_encoder.classes_.tolist()
            
            # Store feature names
            self.feature_names = list(features.columns)
            
            # Prepare features
            X = self._prepare_features(features)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.model.fit(X_scaled, y)
            
            self.is_trained = True
            self.logger.info(f"Trained {self.method} classifier with {len(features)} samples")
            self.logger.info(f"Classes: {self.class_names}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error training attack classifier: {e}")
            return False
    
    def classify_attacks(self, features: pd.DataFrame) -> Tuple[List[str], List[float]]:
        """Classify attacks in the provided features"""
        if not self.is_trained:
            self.logger.error("Model not trained. Call train() first.")
            return [], []
        
        try:
            # Prepare features
            X = self._prepare_features(features)
            X_scaled = self.scaler.transform(X)
            
            # Get predictions and probabilities
            predictions = self.model.predict(X_scaled)
            probabilities = self.model.predict_proba(X_scaled)
            
            # Convert predictions back to class names
            class_names = self.label_encoder.inverse_transform(predictions)
            
            # Get confidence scores (max probability for each prediction)
            confidences = np.max(probabilities, axis=1)
            
            self.logger.info(f"Classified {len(features)} samples")
            return class_names.tolist(), confidences.tolist()
            
        except Exception as e:
            self.logger.error(f"Error classifying attacks: {e}")
            return [], []
    
    def _prepare_features(self, features: pd.DataFrame) -> np.ndarray:
        """Prepare features for the model"""
        # Select only numerical features
        numerical_features = features.select_dtypes(include=[np.number])
        
        # Handle missing values
        numerical_features = numerical_features.fillna(0)
        
        # Convert to numpy array
        return numerical_features.values
    
    def set_confidence_threshold(self, threshold: float):
        """Set confidence threshold for classification"""
        if 0.0 <= threshold <= 1.0:
            self.confidence_threshold = threshold
            self.logger.info(f"Set confidence threshold to {threshold}")
        else:
            self.logger.error(f"Invalid confidence threshold: {threshold}")
    
    def evaluate(self, features: pd.DataFrame, logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate the classifier performance"""
        if not self.is_trained:
            return {"error": "Model not trained"}
        
        try:
            # Extract true labels
            true_labels = []
            for log in logs:
                attack_type = log.get('attack_type', 'none')
                true_labels.append(attack_type)
            
            # Get predictions
            predicted_labels, confidences = self.classify_attacks(features)
            
            if not predicted_labels:
                return {"error": "Failed to get predictions"}
            
            # Calculate metrics
            accuracy = accuracy_score(true_labels, predicted_labels)
            precision, recall, f1, support = precision_recall_fscore_support(
                true_labels, predicted_labels, average='weighted', zero_division=0
            )
            
            # Detailed classification report
            report = classification_report(true_labels, predicted_labels, output_dict=True)
            
            # Class-wise metrics
            class_metrics = {}
            for class_name in self.class_names:
                if class_name in report:
                    class_metrics[class_name] = {
                        "precision": report[class_name]["precision"],
                        "recall": report[class_name]["recall"],
                        "f1_score": report[class_name]["f1-score"],
                        "support": report[class_name]["support"]
                    }
            
            return {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "class_metrics": class_metrics,
                "total_samples": len(features),
                "class_distribution": dict(zip(*np.unique(true_labels, return_counts=True)))
            }
            
        except Exception as e:
            self.logger.error(f"Error evaluating classifier: {e}")
            return {"error": str(e)}
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance (only for Random Forest)"""
        if not self.is_trained or self.method != "random_forest":
            return {}
        
        try:
            if hasattr(self.model, 'feature_importances_'):
                importance = dict(zip(self.feature_names, self.model.feature_importances_))
                return importance
            else:
                return {}
                
        except Exception as e:
            self.logger.error(f"Error getting feature importance: {e}")
            return {}
    
    def save_model(self, filepath: str) -> bool:
        """Save the trained model to file"""
        try:
            if not self.is_trained:
                self.logger.error("Cannot save untrained model")
                return False
            
            model_data = {
                "model": self.model,
                "scaler": self.scaler,
                "label_encoder": self.label_encoder,
                "method": self.method,
                "confidence_threshold": self.confidence_threshold,
                "feature_names": self.feature_names,
                "class_names": self.class_names,
                "is_trained": self.is_trained
            }
            
            joblib.dump(model_data, filepath)
            self.logger.info(f"Saved attack classifier to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load a trained model from file"""
        try:
            model_data = joblib.load(filepath)
            
            self.model = model_data["model"]
            self.scaler = model_data["scaler"]
            self.label_encoder = model_data["label_encoder"]
            self.method = model_data["method"]
            self.confidence_threshold = model_data["confidence_threshold"]
            self.feature_names = model_data["feature_names"]
            self.class_names = model_data["class_names"]
            self.is_trained = model_data["is_trained"]
            
            self.logger.info(f"Loaded attack classifier from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return {
            "method": self.method,
            "confidence_threshold": self.confidence_threshold,
            "is_trained": self.is_trained,
            "feature_count": len(self.feature_names),
            "class_count": len(self.class_names),
            "classes": self.class_names
        }
