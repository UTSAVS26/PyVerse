"""
Threat Detector for HoneypotAI
Main coordinator for all ML-based threat detection models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import structlog

from .feature_extractor import FeatureExtractor
from .anomaly_detector import AnomalyDetector
from .attack_classifier import AttackClassifier

logger = structlog.get_logger(__name__)

class ThreatDetector:
    """Main threat detection system that coordinates all ML models"""
    
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.anomaly_detector = AnomalyDetector()
        self.attack_classifier = AttackClassifier()
        
        self.logger = structlog.get_logger("ml.threat_detector")
        
        # Configuration
        self.anomaly_sensitivity = 0.8
        self.classification_confidence_threshold = 0.9
        self.enable_online_learning = True
        
        # Statistics
        self.detection_stats = {
            "total_detections": 0,
            "anomaly_detections": 0,
            "classification_detections": 0,
            "false_positives": 0,
            "true_positives": 0,
            "start_time": None
        }
    
    def setup_anomaly_detection(self, sensitivity: float = 0.8, method: str = "isolation_forest"):
        """Configure anomaly detection"""
        self.anomaly_sensitivity = max(0.0, min(1.0, sensitivity))
        contamination = 1.0 - self.anomaly_sensitivity
        
        self.anomaly_detector = AnomalyDetector(method=method, contamination=contamination)
        self.logger.info(f"Configured anomaly detection with sensitivity {sensitivity}")
    
    def setup_attack_classification(self, confidence_threshold: float = 0.9):
        """Configure attack classification"""
        self.classification_confidence_threshold = max(0.0, min(1.0, confidence_threshold))
        self.attack_classifier.set_confidence_threshold(self.classification_confidence_threshold)
        self.logger.info(f"Configured attack classification with confidence threshold {confidence_threshold}")
    
    def train_models(self, logs: List[Dict[str, Any]]) -> bool:
        """Train all ML models with provided logs"""
        try:
            if not logs:
                self.logger.warning("No logs provided for training")
                return False
            
            # Extract features
            features = self.feature_extractor.extract_features(logs)
            if features.empty:
                self.logger.error("Failed to extract features from logs")
                return False
            
            # Normalize features
            features_normalized = self.feature_extractor.normalize_features(features)
            
            # Train anomaly detector
            anomaly_success = self.anomaly_detector.train(features_normalized)
            if not anomaly_success:
                self.logger.error("Failed to train anomaly detector")
                return False
            
            # Train attack classifier
            classification_success = self.attack_classifier.train(features_normalized, logs)
            if not classification_success:
                self.logger.error("Failed to train attack classifier")
                return False
            
            self.detection_stats["start_time"] = datetime.now()
            self.logger.info(f"Successfully trained all models with {len(logs)} samples")
            return True
            
        except Exception as e:
            self.logger.error(f"Error training models: {e}")
            return False
    
    def detect_threats(self, logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect threats in the provided logs"""
        try:
            if not logs:
                return []
            
            # Extract features
            features = self.feature_extractor.extract_features(logs)
            if features.empty:
                return []
            
            # Normalize features
            features_normalized = self.feature_extractor.normalize_features(features)
            
            threats = []
            
            # Anomaly detection
            if self.anomaly_detector.is_trained:
                is_anomaly, anomaly_scores = self.anomaly_detector.detect_anomalies(features_normalized)
                
                for i, (log, is_anom, score) in enumerate(zip(logs, is_anomaly, anomaly_scores)):
                    if is_anom:
                        threat = {
                            "timestamp": log.get("timestamp", ""),
                            "source_ip": log.get("source_ip", ""),
                            "service": log.get("service", ""),
                            "threat_type": "anomaly",
                            "confidence": self._normalize_anomaly_score(score),
                            "details": {
                                "anomaly_score": float(score),
                                "detection_method": "unsupervised"
                            }
                        }
                        threats.append(threat)
                        self.detection_stats["anomaly_detections"] += 1
            
            # Attack classification
            if self.attack_classifier.is_trained:
                classifications, confidences = self.attack_classifier.classify_attacks(features_normalized)
                
                for i, (log, attack_type, confidence) in enumerate(zip(logs, classifications, confidences)):
                    if attack_type != "none" and confidence >= self.classification_confidence_threshold:
                        threat = {
                            "timestamp": log.get("timestamp", ""),
                            "source_ip": log.get("source_ip", ""),
                            "service": log.get("service", ""),
                            "threat_type": "classified_attack",
                            "confidence": float(confidence),
                            "details": {
                                "attack_type": attack_type,
                                "detection_method": "supervised"
                            }
                        }
                        threats.append(threat)
                        self.detection_stats["classification_detections"] += 1
            
            # Update statistics
            self.detection_stats["total_detections"] += len(threats)
            
            self.logger.info(f"Detected {len(threats)} threats from {len(logs)} logs")
            return threats
            
        except Exception as e:
            self.logger.error(f"Error detecting threats: {e}")
            return []
    
    def _normalize_anomaly_score(self, score: float) -> float:
        """Normalize anomaly score to 0-1 range"""
        # Anomaly scores are typically negative for anomalies
        # We normalize them to 0-1 where 1 is most anomalous
        if score < 0:
            return min(1.0, abs(score) / 2.0)  # Normalize negative scores
        else:
            return max(0.0, 1.0 - score)  # Invert positive scores
    
    def start_detection(self):
        """Start real-time threat detection"""
        self.logger.info("Starting real-time threat detection")
        # In a real implementation, this would start a background thread
        # that continuously monitors for new logs
    
    def stop_detection(self):
        """Stop real-time threat detection"""
        self.logger.info("Stopping real-time threat detection")
        # In a real implementation, this would stop the background thread
    
    def get_detection_stats(self) -> Dict[str, Any]:
        """Get threat detection statistics"""
        stats = self.detection_stats.copy()
        
        # Calculate uptime
        if stats["start_time"]:
            stats["uptime"] = (datetime.now() - stats["start_time"]).total_seconds()
        
        # Add model status
        stats["anomaly_detector_trained"] = self.anomaly_detector.is_trained
        stats["attack_classifier_trained"] = self.attack_classifier.is_trained
        
        # Add model performance metrics
        if self.anomaly_detector.is_trained:
            stats["anomaly_detector_info"] = self.anomaly_detector.get_model_info()
        
        if self.attack_classifier.is_trained:
            stats["attack_classifier_info"] = self.attack_classifier.get_model_info()
        
        return stats
    
    def evaluate_performance(self, test_logs: List[Dict[str, Any]], 
                           true_labels: Optional[List[str]] = None) -> Dict[str, Any]:
        """Evaluate the performance of threat detection models"""
        try:
            results = {}
            
            # Extract features
            features = self.feature_extractor.extract_features(test_logs)
            if features.empty:
                return {"error": "Failed to extract features"}
            
            features_normalized = self.feature_extractor.normalize_features(features)
            
            # Evaluate anomaly detector
            if self.anomaly_detector.is_trained:
                anomaly_results = self.anomaly_detector.evaluate(features_normalized)
                results["anomaly_detector"] = anomaly_results
            
            # Evaluate attack classifier
            if self.attack_classifier.is_trained:
                classification_results = self.attack_classifier.evaluate(features_normalized, test_logs)
                results["attack_classifier"] = classification_results
            
            # Overall evaluation
            detected_threats = self.detect_threats(test_logs)
            results["overall"] = {
                "total_threats_detected": len(detected_threats),
                "total_samples": len(test_logs),
                "detection_rate": len(detected_threats) / len(test_logs) if test_logs else 0
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error evaluating performance: {e}")
            return {"error": str(e)}
    
    def save_models(self, base_path: str) -> bool:
        """Save all trained models"""
        try:
            success = True
            
            # Save anomaly detector
            if self.anomaly_detector.is_trained:
                anomaly_path = f"{base_path}/anomaly_detector.pkl"
                if not self.anomaly_detector.save_model(anomaly_path):
                    success = False
            
            # Save attack classifier
            if self.attack_classifier.is_trained:
                classifier_path = f"{base_path}/attack_classifier.pkl"
                if not self.attack_classifier.save_model(classifier_path):
                    success = False
            
            if success:
                self.logger.info(f"Saved all models to {base_path}")
            else:
                self.logger.error("Failed to save some models")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error saving models: {e}")
            return False
    
    def load_models(self, base_path: str) -> bool:
        """Load all trained models"""
        try:
            success = True
            
            # Load anomaly detector
            anomaly_path = f"{base_path}/anomaly_detector.pkl"
            if not self.anomaly_detector.load_model(anomaly_path):
                self.logger.warning("Failed to load anomaly detector")
                success = False
            
            # Load attack classifier
            classifier_path = f"{base_path}/attack_classifier.pkl"
            if not self.attack_classifier.load_model(classifier_path):
                self.logger.warning("Failed to load attack classifier")
                success = False
            
            if success:
                self.logger.info(f"Loaded all models from {base_path}")
            else:
                self.logger.warning("Some models failed to load")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            return False
    
    def get_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """Get feature importance for all models"""
        importance = {}
        
        # Anomaly detector feature importance
        if self.anomaly_detector.is_trained:
            importance["anomaly_detector"] = self.anomaly_detector.get_feature_importance()
        
        # Attack classifier feature importance
        if self.attack_classifier.is_trained:
            importance["attack_classifier"] = self.attack_classifier.get_feature_importance()
        
        return importance
