"""
Anomaly Detector for HoneypotAI
Uses unsupervised learning to detect unknown attack patterns and anomalies
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
import joblib
import structlog

logger = structlog.get_logger(__name__)

class AnomalyDetector:
    """Anomaly detection using unsupervised learning methods"""
    
    def __init__(self, method: str = "isolation_forest", contamination: float = 0.1):
        self.method = method
        self.contamination = contamination
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []
        
        self.logger = structlog.get_logger("ml.anomaly_detector")
        
        # Initialize model based on method
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the anomaly detection model"""
        if self.method == "isolation_forest":
            self.model = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=100,
                max_samples='auto'
            )
        elif self.method == "one_class_svm":
            self.model = OneClassSVM(
                kernel='rbf',
                nu=self.contamination,
                gamma='scale'
            )
        else:
            raise ValueError(f"Unknown anomaly detection method: {self.method}")
    
    def train(self, features: pd.DataFrame) -> bool:
        """Train the anomaly detection model"""
        try:
            if features.empty:
                self.logger.warning("No features provided for training")
                return False
            
            # Store feature names
            self.feature_names = list(features.columns)
            
            # Prepare features for training
            X = self._prepare_features(features)
            
            # Fit scaler
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            if self.method == "one_class_svm":
                # OneClassSVM expects normal data (no anomalies)
                # We'll use all data but adjust the nu parameter
                self.model.fit(X_scaled)
            else:
                # IsolationForest can handle mixed data
                self.model.fit(X_scaled)
            
            self.is_trained = True
            self.logger.info(f"Trained {self.method} anomaly detector with {len(features)} samples")
            return True
            
        except Exception as e:
            self.logger.error(f"Error training anomaly detector: {e}")
            return False
    
    def detect_anomalies(self, features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Detect anomalies in the provided features"""
        if not self.is_trained:
            self.logger.error("Model not trained. Call train() first.")
            return np.array([]), np.array([])
        
        try:
            # Prepare features
            X = self._prepare_features(features)
            X_scaled = self.scaler.transform(X)
            
            # Predict anomalies
            if self.method == "isolation_forest":
                # IsolationForest returns -1 for anomalies, 1 for normal
                predictions = self.model.predict(X_scaled)
                anomaly_scores = self.model.decision_function(X_scaled)
            else:
                # OneClassSVM returns -1 for anomalies, 1 for normal
                predictions = self.model.predict(X_scaled)
                anomaly_scores = self.model.decision_function(X_scaled)
            
            # Convert to boolean (True for anomalies)
            is_anomaly = predictions == -1
            
            self.logger.info(f"Detected {np.sum(is_anomaly)} anomalies out of {len(features)} samples")
            return is_anomaly, anomaly_scores
            
        except Exception as e:
            self.logger.error(f"Error detecting anomalies: {e}")
            return np.array([]), np.array([])
    
    def _prepare_features(self, features: pd.DataFrame) -> np.ndarray:
        """Prepare features for the model"""
        # Select only numerical features
        numerical_features = features.select_dtypes(include=[np.number])
        
        # Handle missing values
        numerical_features = numerical_features.fillna(0)
        
        # Convert to numpy array
        return numerical_features.values
    
    def evaluate(self, features: pd.DataFrame, labels: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Evaluate the anomaly detector performance"""
        if not self.is_trained:
            return {"error": "Model not trained"}
        
        try:
            # Detect anomalies
            is_anomaly, scores = self.detect_anomalies(features)
            
            if labels is not None:
                # Calculate metrics if true labels are available
                # Note: labels should be 1 for normal, -1 for anomaly
                if len(labels) == len(is_anomaly):
                    # Convert predictions to same format as labels
                    predictions = np.where(is_anomaly, -1, 1)
                    
                    # Calculate metrics
                    precision = precision_score(labels, predictions, pos_label=-1, zero_division=0)
                    recall = recall_score(labels, predictions, pos_label=-1, zero_division=0)
                    f1 = f1_score(labels, predictions, pos_label=-1, zero_division=0)
                    
                    return {
                        "precision": precision,
                        "recall": recall,
                        "f1_score": f1,
                        "anomaly_count": np.sum(is_anomaly),
                        "total_samples": len(features),
                        "anomaly_rate": np.mean(is_anomaly)
                    }
            
            # Return basic statistics if no labels
            return {
                "anomaly_count": np.sum(is_anomaly),
                "total_samples": len(features),
                "anomaly_rate": np.mean(is_anomaly),
                "mean_score": np.mean(scores),
                "std_score": np.std(scores)
            }
            
        except Exception as e:
            self.logger.error(f"Error evaluating anomaly detector: {e}")
            return {"error": str(e)}
    
    def get_anomaly_threshold(self) -> float:
        """Get the current anomaly threshold"""
        if not self.is_trained:
            return 0.0
        
        # For IsolationForest, we can estimate threshold from contamination
        if self.method == "isolation_forest":
            return -0.5  # Approximate threshold
        else:
            return 0.0  # OneClassSVM doesn't have a simple threshold
    
    def set_contamination(self, contamination: float):
        """Update contamination parameter and retrain if necessary"""
        if 0.0 <= contamination <= 1.0:
            self.contamination = contamination
            self._initialize_model()
            self.is_trained = False  # Need to retrain
            self.logger.info(f"Updated contamination to {contamination}")
        else:
            self.logger.error(f"Invalid contamination value: {contamination}")
    
    def save_model(self, filepath: str) -> bool:
        """Save the trained model to file"""
        try:
            if not self.is_trained:
                self.logger.error("Cannot save untrained model")
                return False
            
            model_data = {
                "model": self.model,
                "scaler": self.scaler,
                "method": self.method,
                "contamination": self.contamination,
                "feature_names": self.feature_names,
                "is_trained": self.is_trained
            }
            
            joblib.dump(model_data, filepath)
            self.logger.info(f"Saved anomaly detector to {filepath}")
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
            self.method = model_data["method"]
            self.contamination = model_data["contamination"]
            self.feature_names = model_data["feature_names"]
            self.is_trained = model_data["is_trained"]
            
            self.logger.info(f"Loaded anomaly detector from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return False
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance (only for IsolationForest)"""
        if not self.is_trained or self.method != "isolation_forest":
            return {}
        
        try:
            # IsolationForest doesn't provide direct feature importance
            # We can estimate it by measuring how much each feature affects anomaly scores
            if not self.feature_names:
                return {}
            
            # Create baseline features (zeros)
            baseline = np.zeros((1, len(self.feature_names)))
            baseline_scaled = self.scaler.transform(baseline)
            baseline_score = self.model.decision_function(baseline_scaled)[0]
            
            # Calculate importance for each feature
            importance = {}
            for i, feature_name in enumerate(self.feature_names):
                # Create features with this feature set to 1
                test_features = np.zeros((1, len(self.feature_names)))
                test_features[0, i] = 1.0
                test_scaled = self.scaler.transform(test_features)
                test_score = self.model.decision_function(test_scaled)[0]
                
                # Importance is the change in score
                importance[feature_name] = abs(test_score - baseline_score)
            
            return importance
            
        except Exception as e:
            self.logger.error(f"Error calculating feature importance: {e}")
            return {}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return {
            "method": self.method,
            "contamination": self.contamination,
            "is_trained": self.is_trained,
            "feature_count": len(self.feature_names),
            "feature_names": self.feature_names
        }
