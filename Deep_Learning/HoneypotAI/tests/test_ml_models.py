"""
Tests for HoneypotAI ML Module
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml import ThreatDetector, FeatureExtractor, AnomalyDetector, AttackClassifier, OnlineTrainer

class TestFeatureExtractor:
    """Test feature extraction functionality"""
    
    def test_feature_extractor_initialization(self):
        """Test feature extractor initialization"""
        extractor = FeatureExtractor()
        
        assert "basic_features" in extractor.__dict__
        assert "network_features" in extractor.__dict__
        assert "temporal_features" in extractor.__dict__
        assert "behavioral_features" in extractor.__dict__
        assert "security_features" in extractor.__dict__
    
    def test_extract_features_empty_logs(self):
        """Test feature extraction with empty logs"""
        extractor = FeatureExtractor()
        
        features = extractor.extract_features([])
        assert features.empty
    
    def test_extract_features_single_log(self):
        """Test feature extraction with single log"""
        extractor = FeatureExtractor()
        
        log = {
            "timestamp": "2023-01-01T12:00:00",
            "source_ip": "192.168.1.1",
            "source_port": 12345,
            "service": "ssh",
            "payload_size": 100,
            "connection_duration": 1.5,
            "success": True,
            "attack_type": "none",
            "confidence": 0.0
        }
        
        features = extractor.extract_features([log])
        assert not features.empty
        assert len(features) == 1
        assert "payload_size" in features.columns
        assert "source_port" in features.columns
        assert "service_ssh" in features.columns
    
    def test_extract_basic_features(self):
        """Test basic feature extraction"""
        extractor = FeatureExtractor()
        
        log = {
            "payload_size": 100,
            "connection_duration": 1.5,
            "success": True
        }
        
        features = extractor._extract_basic_features(log)
        
        assert features["payload_size"] == 100
        assert features["connection_duration"] == 1.5
        assert features["success"] == 1
        assert "payload_size_log" in features
        assert "connection_duration_log" in features
    
    def test_extract_network_features(self):
        """Test network feature extraction"""
        extractor = FeatureExtractor()
        
        log = {
            "source_port": 22,
            "service": "ssh"
        }
        
        features = extractor._extract_network_features(log)
        
        assert features["source_port"] == 22
        assert features["is_privileged_port"] == 1
        assert features["is_common_port"] == 1
        assert features["service_ssh"] == 1
        assert features["service_http"] == 0
    
    def test_extract_temporal_features(self):
        """Test temporal feature extraction"""
        extractor = FeatureExtractor()
        
        log = {
            "timestamp": "2023-01-01T12:00:00"
        }
        
        features = extractor._extract_temporal_features(log)
        
        assert "hour_of_day" in features
        assert "day_of_week" in features
        assert "is_weekend" in features
        assert "hour_sin" in features
        assert "hour_cos" in features
        assert "day_sin" in features
        assert "day_cos" in features
    
    def test_extract_security_features(self):
        """Test security feature extraction"""
        extractor = FeatureExtractor()
        
        log = {
            "attack_type": "sql_injection",
            "confidence": 0.9
        }
        
        features = extractor._extract_security_features(log)
        
        assert features["attack_sql_injection"] == 1
        assert features["attack_xss"] == 0
        assert features["confidence"] == 0.9
        assert features["is_attack"] == 1
        assert "suspicious_patterns" in features
    
    def test_calculate_entropy(self):
        """Test entropy calculation"""
        extractor = FeatureExtractor()
        
        # Test with simple string
        entropy = extractor._calculate_entropy("aaaa")
        assert entropy == 0.0
        
        # Test with diverse string
        entropy = extractor._calculate_entropy("abcd")
        assert entropy > 0.0
        
        # Test with empty string
        entropy = extractor._calculate_entropy("")
        assert entropy == 0.0
    
    def test_normalize_features(self):
        """Test feature normalization"""
        extractor = FeatureExtractor()
        
        # Create test features
        features = pd.DataFrame({
            "payload_size": [100, 200, 300],
            "connection_duration": [1.0, 2.0, 3.0],
            "success": [1, 0, 1]
        })
        
        normalized = extractor.normalize_features(features)
        
        assert not normalized.empty
        assert "payload_size" in normalized.columns
        assert "connection_duration" in normalized.columns
        assert "success" in normalized.columns

class TestAnomalyDetector:
    """Test anomaly detection functionality"""
    
    def test_anomaly_detector_initialization(self):
        """Test anomaly detector initialization"""
        detector = AnomalyDetector()
        
        assert detector.method == "isolation_forest"
        assert detector.contamination == 0.1
        assert detector.model is not None
        assert not detector.is_trained
    
    def test_anomaly_detector_initialization_one_class_svm(self):
        """Test anomaly detector initialization with OneClassSVM"""
        detector = AnomalyDetector(method="one_class_svm")
        
        assert detector.method == "one_class_svm"
        assert detector.contamination == 0.1
        assert detector.model is not None
    
    def test_anomaly_detector_invalid_method(self):
        """Test anomaly detector with invalid method"""
        with pytest.raises(ValueError):
            AnomalyDetector(method="invalid_method")
    
    def test_train_anomaly_detector(self):
        """Test anomaly detector training"""
        detector = AnomalyDetector()
        
        # Create test features
        features = pd.DataFrame({
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
            "feature3": np.random.randn(100)
        })
        
        success = detector.train(features)
        assert success
        assert detector.is_trained
        assert len(detector.feature_names) == 3
    
    def test_train_anomaly_detector_empty_features(self):
        """Test anomaly detector training with empty features"""
        detector = AnomalyDetector()
        
        features = pd.DataFrame()
        success = detector.train(features)
        assert not success
    
    def test_detect_anomalies(self):
        """Test anomaly detection"""
        detector = AnomalyDetector()
        
        # Train the detector first
        features = pd.DataFrame({
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100)
        })
        detector.train(features)
        
        # Test detection
        test_features = pd.DataFrame({
            "feature1": np.random.randn(10),
            "feature2": np.random.randn(10)
        })
        
        is_anomaly, scores = detector.detect_anomalies(test_features)
        
        assert len(is_anomaly) == 10
        assert len(scores) == 10
        assert isinstance(is_anomaly, np.ndarray)
        assert isinstance(scores, np.ndarray)
    
    def test_detect_anomalies_not_trained(self):
        """Test anomaly detection without training"""
        detector = AnomalyDetector()
        
        test_features = pd.DataFrame({
            "feature1": np.random.randn(10),
            "feature2": np.random.randn(10)
        })
        
        is_anomaly, scores = detector.detect_anomalies(test_features)
        assert len(is_anomaly) == 0
        assert len(scores) == 0
    
    def test_evaluate_anomaly_detector(self):
        """Test anomaly detector evaluation"""
        detector = AnomalyDetector()
        
        # Train the detector
        features = pd.DataFrame({
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100)
        })
        detector.train(features)
        
        # Evaluate without labels
        results = detector.evaluate(features)
        assert "anomaly_count" in results
        assert "total_samples" in results
        assert "anomaly_rate" in results
    
    def test_set_contamination(self):
        """Test setting contamination parameter"""
        detector = AnomalyDetector()
        
        detector.set_contamination(0.2)
        assert detector.contamination == 0.2
        assert not detector.is_trained  # Should reset training state
    
    def test_set_contamination_invalid(self):
        """Test setting invalid contamination parameter"""
        detector = AnomalyDetector()
        
        original_contamination = detector.contamination
        detector.set_contamination(1.5)  # Invalid value
        assert detector.contamination == original_contamination

class TestAttackClassifier:
    """Test attack classification functionality"""
    
    def test_attack_classifier_initialization(self):
        """Test attack classifier initialization"""
        classifier = AttackClassifier()
        
        assert classifier.method == "random_forest"
        assert classifier.confidence_threshold == 0.9
        assert classifier.model is not None
        assert not classifier.is_trained
    
    def test_attack_classifier_initialization_logistic_regression(self):
        """Test attack classifier initialization with logistic regression"""
        classifier = AttackClassifier(method="logistic_regression")
        
        assert classifier.method == "logistic_regression"
        assert classifier.model is not None
    
    def test_attack_classifier_initialization_svm(self):
        """Test attack classifier initialization with SVM"""
        classifier = AttackClassifier(method="svm")
        
        assert classifier.method == "svm"
        assert classifier.model is not None
    
    def test_attack_classifier_invalid_method(self):
        """Test attack classifier with invalid method"""
        with pytest.raises(ValueError):
            AttackClassifier(method="invalid_method")
    
    def test_train_attack_classifier(self):
        """Test attack classifier training"""
        classifier = AttackClassifier()
        
        # Create test features and logs
        features = pd.DataFrame({
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100)
        })
        
        logs = [
            {"attack_type": "sql_injection"},
            {"attack_type": "xss"},
            {"attack_type": "none"}
        ] * 33 + [{"attack_type": "sql_injection"}]  # 100 total logs
        
        success = classifier.train(features, logs)
        assert success
        assert classifier.is_trained
        assert len(classifier.class_names) > 0
    
    def test_train_attack_classifier_empty_data(self):
        """Test attack classifier training with empty data"""
        classifier = AttackClassifier()
        
        features = pd.DataFrame()
        logs = []
        
        success = classifier.train(features, logs)
        assert not success
    
    def test_classify_attacks(self):
        """Test attack classification"""
        classifier = AttackClassifier()
        
        # Train the classifier first
        features = pd.DataFrame({
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100)
        })
        
        logs = [
            {"attack_type": "sql_injection"},
            {"attack_type": "xss"},
            {"attack_type": "none"}
        ] * 33 + [{"attack_type": "sql_injection"}]
        
        classifier.train(features, logs)
        
        # Test classification
        test_features = pd.DataFrame({
            "feature1": np.random.randn(10),
            "feature2": np.random.randn(10)
        })
        
        classifications, confidences = classifier.classify_attacks(test_features)
        
        assert len(classifications) == 10
        assert len(confidences) == 10
        assert isinstance(classifications, list)
        assert isinstance(confidences, list)
    
    def test_classify_attacks_not_trained(self):
        """Test attack classification without training"""
        classifier = AttackClassifier()
        
        test_features = pd.DataFrame({
            "feature1": np.random.randn(10),
            "feature2": np.random.randn(10)
        })
        
        classifications, confidences = classifier.classify_attacks(test_features)
        assert len(classifications) == 0
        assert len(confidences) == 0
    
    def test_set_confidence_threshold(self):
        """Test setting confidence threshold"""
        classifier = AttackClassifier()
        
        classifier.set_confidence_threshold(0.8)
        assert classifier.confidence_threshold == 0.8
    
    def test_set_confidence_threshold_invalid(self):
        """Test setting invalid confidence threshold"""
        classifier = AttackClassifier()
        
        original_threshold = classifier.confidence_threshold
        classifier.set_confidence_threshold(1.5)  # Invalid value
        assert classifier.confidence_threshold == original_threshold
    
    def test_evaluate_attack_classifier(self):
        """Test attack classifier evaluation"""
        classifier = AttackClassifier()
        
        # Train the classifier
        features = pd.DataFrame({
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100)
        })
        
        logs = [
            {"attack_type": "sql_injection"},
            {"attack_type": "xss"},
            {"attack_type": "none"}
        ] * 33 + [{"attack_type": "sql_injection"}]
        
        classifier.train(features, logs)
        
        # Evaluate
        results = classifier.evaluate(features, logs)
        assert "accuracy" in results
        assert "precision" in results
        assert "recall" in results
        assert "f1_score" in results

class TestThreatDetector:
    """Test threat detector functionality"""
    
    def test_threat_detector_initialization(self):
        """Test threat detector initialization"""
        detector = ThreatDetector()
        
        assert detector.anomaly_sensitivity == 0.8
        assert detector.classification_confidence_threshold == 0.9
        assert detector.enable_online_learning is True
        assert detector.feature_extractor is not None
        assert detector.anomaly_detector is not None
        assert detector.attack_classifier is not None
    
    def test_setup_anomaly_detection(self):
        """Test anomaly detection setup"""
        detector = ThreatDetector()
        
        detector.setup_anomaly_detection(sensitivity=0.7, method="one_class_svm")
        assert detector.anomaly_sensitivity == 0.7
        assert detector.anomaly_detector.method == "one_class_svm"
    
    def test_setup_attack_classification(self):
        """Test attack classification setup"""
        detector = ThreatDetector()
        
        detector.setup_attack_classification(confidence_threshold=0.8)
        assert detector.classification_confidence_threshold == 0.8
    
    def test_train_models(self):
        """Test model training"""
        detector = ThreatDetector()
        
        # Create test logs
        logs = [
            {
                "timestamp": "2023-01-01T12:00:00",
                "source_ip": "192.168.1.1",
                "source_port": 12345,
                "service": "ssh",
                "payload_size": 100,
                "connection_duration": 1.5,
                "success": True,
                "attack_type": "sql_injection",
                "confidence": 0.9
            }
        ] * 50  # 50 logs for training
        
        success = detector.train_models(logs)
        assert success
    
    def test_train_models_empty_logs(self):
        """Test model training with empty logs"""
        detector = ThreatDetector()
        
        success = detector.train_models([])
        assert not success
    
    def test_detect_threats(self):
        """Test threat detection"""
        detector = ThreatDetector()
        
        # Train models first
        logs = [
            {
                "timestamp": "2023-01-01T12:00:00",
                "source_ip": "192.168.1.1",
                "source_port": 12345,
                "service": "ssh",
                "payload_size": 100,
                "connection_duration": 1.5,
                "success": True,
                "attack_type": "sql_injection",
                "confidence": 0.9
            }
        ] * 50
        
        detector.train_models(logs)
        
        # Test threat detection
        test_logs = [
            {
                "timestamp": "2023-01-01T12:00:00",
                "source_ip": "192.168.1.2",
                "source_port": 12346,
                "service": "http",
                "payload_size": 200,
                "connection_duration": 2.0,
                "success": False,
                "attack_type": "xss",
                "confidence": 0.8
            }
        ]
        
        threats = detector.detect_threats(test_logs)
        assert isinstance(threats, list)
    
    def test_detect_threats_empty_logs(self):
        """Test threat detection with empty logs"""
        detector = ThreatDetector()
        
        threats = detector.detect_threats([])
        assert len(threats) == 0
    
    def test_get_detection_stats(self):
        """Test getting detection statistics"""
        detector = ThreatDetector()
        
        stats = detector.get_detection_stats()
        assert "total_detections" in stats
        assert "anomaly_detections" in stats
        assert "classification_detections" in stats
        assert "anomaly_detector_trained" in stats
        assert "attack_classifier_trained" in stats

class TestOnlineTrainer:
    """Test online training functionality"""
    
    def test_online_trainer_initialization(self):
        """Test online trainer initialization"""
        threat_detector = ThreatDetector()
        trainer = OnlineTrainer(threat_detector)
        
        assert trainer.batch_size == 100
        assert trainer.retrain_interval == 3600
        assert trainer.min_samples_for_retrain == 50
        assert not trainer.is_running
        assert len(trainer.sample_buffer) == 0
    
    def test_add_sample(self):
        """Test adding samples"""
        threat_detector = ThreatDetector()
        trainer = OnlineTrainer(threat_detector)
        
        sample = {
            "timestamp": "2023-01-01T12:00:00",
            "source_ip": "192.168.1.1",
            "service": "ssh"
        }
        
        trainer.add_sample(sample)
        assert trainer.training_stats["total_samples_processed"] == 1
    
    def test_add_samples_batch(self):
        """Test adding multiple samples"""
        threat_detector = ThreatDetector()
        trainer = OnlineTrainer(threat_detector)
        
        samples = [
            {"timestamp": "2023-01-01T12:00:00", "source_ip": "192.168.1.1"},
            {"timestamp": "2023-01-01T12:01:00", "source_ip": "192.168.1.2"}
        ]
        
        trainer.add_samples_batch(samples)
        assert trainer.training_stats["total_samples_processed"] == 2
    
    def test_get_training_stats(self):
        """Test getting training statistics"""
        threat_detector = ThreatDetector()
        trainer = OnlineTrainer(threat_detector)
        
        stats = trainer.get_training_stats()
        assert "total_samples_processed" in stats
        assert "retrain_count" in stats
        assert "is_running" in stats
        assert "queue_size" in stats
        assert "buffer_size" in stats
    
    def test_set_batch_size(self):
        """Test setting batch size"""
        threat_detector = ThreatDetector()
        trainer = OnlineTrainer(threat_detector)
        
        trainer.set_batch_size(200)
        assert trainer.batch_size == 200
    
    def test_set_retrain_interval(self):
        """Test setting retrain interval"""
        threat_detector = ThreatDetector()
        trainer = OnlineTrainer(threat_detector)
        
        trainer.set_retrain_interval(1800)
        assert trainer.retrain_interval == 1800
    
    def test_set_min_samples_for_retrain(self):
        """Test setting minimum samples for retrain"""
        threat_detector = ThreatDetector()
        trainer = OnlineTrainer(threat_detector)
        
        trainer.set_min_samples_for_retrain(100)
        assert trainer.min_samples_for_retrain == 100

if __name__ == "__main__":
    pytest.main([__file__])
