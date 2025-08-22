"""
Tests for the prosody classifier module.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emotion.prosody_classifier import RuleBasedClassifier, MLClassifier, HybridClassifier


class TestRuleBasedClassifier:
    """Test cases for RuleBasedClassifier class."""
    
    def setup_method(self):
        """Setup method called before each test."""
        self.classifier = RuleBasedClassifier()
    
    def test_initialization(self):
        """Test RuleBasedClassifier initialization."""
        assert len(self.classifier.emotions) == 7
        assert 'happy' in self.classifier.emotions
        assert 'sad' in self.classifier.emotions
        assert 'angry' in self.classifier.emotions
        assert 'calm' in self.classifier.emotions
        assert 'excited' in self.classifier.emotions
        assert 'tired' in self.classifier.emotions
        assert 'neutral' in self.classifier.emotions
        
        assert len(self.classifier.rules) > 0
    
    def test_classify_happy(self):
        """Test classification of happy emotion."""
        features = {
            'pitch_mean': 200,  # High pitch
            'energy_mean': 0.2,  # High energy
            'tempo': 150,  # Fast tempo
            'pitch_variability': 0.25  # High variability
        }
        
        emotion, confidence = self.classifier.classify(features)
        
        assert emotion == 'happy'
        assert confidence > 0.5
    
    def test_classify_sad(self):
        """Test classification of sad emotion."""
        features = {
            'pitch_mean': 100,  # Low pitch
            'energy_mean': 0.05,  # Low energy
            'tempo': 80,  # Slow tempo
            'pitch_variability': 0.05  # Low variability
        }
        
        emotion, confidence = self.classifier.classify(features)
        
        assert emotion == 'sad'
        assert confidence > 0.5
    
    def test_classify_angry(self):
        """Test classification of angry emotion."""
        features = {
            'pitch_mean': 220,  # Very high pitch
            'energy_mean': 0.25,  # High energy
            'pitch_variability': 0.35,  # High variability
            'energy_variability': 0.9  # High energy variability
        }
        
        emotion, confidence = self.classifier.classify(features)
        
        assert emotion == 'angry'
        assert confidence > 0.5
    
    def test_classify_calm(self):
        """Test classification of calm emotion."""
        features = {
            'pitch_mean': 150,  # Medium pitch
            'energy_mean': 0.08,  # Low energy
            'pitch_variability': 0.1,  # Low variability
            'energy_variability': 0.3  # Low energy variability
        }
        
        emotion, confidence = self.classifier.classify(features)
        
        assert emotion == 'calm'
        assert confidence > 0.5
    
    def test_classify_excited(self):
        """Test classification of excited emotion."""
        features = {
            'pitch_mean': 210,  # High pitch
            'energy_mean': 0.22,  # High energy
            'tempo': 160,  # Fast tempo
            'pitch_variability': 0.3  # High variability
        }
        
        emotion, confidence = self.classifier.classify(features)
        
        assert emotion == 'excited'
        assert confidence > 0.5
    
    def test_classify_tired(self):
        """Test classification of tired emotion."""
        features = {
            'pitch_mean': 90,  # Very low pitch
            'energy_mean': 0.03,  # Very low energy
            'tempo': 70,  # Very slow tempo
            'pitch_variability': 0.05  # Very low variability
        }
        
        emotion, confidence = self.classifier.classify(features)
        
        assert emotion == 'tired'
        assert confidence > 0.5
    
    def test_classify_neutral(self):
        """Test classification when no rules match."""
        features = {
            'pitch_mean': 150,  # Medium pitch
            'energy_mean': 0.1,  # Medium energy
            'tempo': 120,  # Medium tempo
            'pitch_variability': 0.15  # Medium variability
        }
        
        emotion, confidence = self.classifier.classify(features)
        
        assert emotion == 'neutral'
        assert confidence == 0.5
    
    def test_get_emotion_probabilities(self):
        """Test getting emotion probabilities."""
        features = {
            'pitch_mean': 200,
            'energy_mean': 0.2,
            'tempo': 150,
            'pitch_variability': 0.25
        }
        
        probabilities = self.classifier.get_emotion_probabilities(features)
        
        assert isinstance(probabilities, dict)
        assert len(probabilities) == len(self.classifier.emotions)
        
        # All probabilities should sum to 1.0
        total_prob = sum(probabilities.values())
        assert abs(total_prob - 1.0) < 0.001
        
        # Happy should have highest probability
        assert probabilities['happy'] > 0.5
    
    def test_edge_cases(self):
        """Test edge cases."""
        # Test with empty features
        empty_features = {}
        emotion, confidence = self.classifier.classify(empty_features)
        
        assert emotion == 'neutral'
        assert confidence == 0.5
        
        # Test with missing features
        partial_features = {'pitch_mean': 150}
        emotion, confidence = self.classifier.classify(partial_features)
        
        assert emotion == 'neutral'
        assert confidence == 0.5
    
    def test_rule_conditions(self):
        """Test individual rule conditions."""
        # Test happy rule conditions
        happy_rule = next(rule for rule in self.classifier.rules if rule['emotion'] == 'happy')
        
        # Test with features that should match
        happy_features = {
            'pitch_mean': 200,
            'energy_mean': 0.2,
            'tempo': 150,
            'pitch_variability': 0.25
        }
        
        conditions_met = sum(1 for condition in happy_rule['conditions'] if condition(happy_features))
        total_conditions = len(happy_rule['conditions'])
        
        assert conditions_met >= total_conditions * 0.75
    
    def test_confidence_calculation(self):
        """Test confidence calculation based on conditions met."""
        features = {
            'pitch_mean': 200,
            'energy_mean': 0.2,
            'tempo': 150,
            'pitch_variability': 0.25
        }
        
        emotion, confidence = self.classifier.classify(features)
        
        # Confidence should be between 0 and 1
        assert 0 <= confidence <= 1
        
        # Should be proportional to conditions met
        assert confidence > 0.5


class TestMLClassifier:
    """Test cases for MLClassifier class."""
    
    def setup_method(self):
        """Setup method called before each test."""
        self.classifier = MLClassifier()
    
    def test_initialization(self):
        """Test MLClassifier initialization."""
        assert self.classifier.model is None
        assert self.classifier.feature_names is None
        assert len(self.classifier.emotions) == 7
    
    def test_train(self):
        """Test training the ML classifier."""
        # Create mock training data
        features_list = [
            {'pitch_mean': 200, 'energy_mean': 0.2, 'tempo': 150},
            {'pitch_mean': 100, 'energy_mean': 0.05, 'tempo': 80},
            {'pitch_mean': 150, 'energy_mean': 0.1, 'tempo': 120}
        ]
        labels = ['happy', 'sad', 'neutral']
        
        self.classifier.train(features_list, labels, model_type='decision_tree')
        
        assert self.classifier.model is not None
        assert self.classifier.feature_names is not None
        assert len(self.classifier.feature_names) == 3
    
    def test_train_random_forest(self):
        """Test training with random forest."""
        features_list = [
            {'pitch_mean': 200, 'energy_mean': 0.2},
            {'pitch_mean': 100, 'energy_mean': 0.05}
        ]
        labels = ['happy', 'sad']
        
        self.classifier.train(features_list, labels, model_type='random_forest')
        
        assert self.classifier.model is not None
    
    def test_train_invalid_model_type(self):
        """Test training with invalid model type."""
        features_list = [{'pitch_mean': 200}]
        labels = ['happy']
        
        with pytest.raises(ValueError):
            self.classifier.train(features_list, labels, model_type='invalid')
    
    def test_train_empty_data(self):
        """Test training with empty data."""
        with pytest.raises(ValueError):
            self.classifier.train([], [])
    
    def test_classify_without_training(self):
        """Test classification without training."""
        features = {'pitch_mean': 200, 'energy_mean': 0.2}
        
        with pytest.raises(ValueError):
            self.classifier.classify(features)
    
    def test_classify_with_training(self):
        """Test classification after training."""
        # Train the classifier
        features_list = [
            {'pitch_mean': 200, 'energy_mean': 0.2},
            {'pitch_mean': 100, 'energy_mean': 0.05}
        ]
        labels = ['happy', 'sad']
        
        self.classifier.train(features_list, labels)
        
        # Test classification
        features = {'pitch_mean': 180, 'energy_mean': 0.15}
        emotion, confidence = self.classifier.classify(features)
        
        assert emotion in ['happy', 'sad']
        assert 0 <= confidence <= 1
    
    def test_get_emotion_probabilities(self):
        """Test getting emotion probabilities."""
        # Train the classifier
        features_list = [
            {'pitch_mean': 200, 'energy_mean': 0.2},
            {'pitch_mean': 100, 'energy_mean': 0.05}
        ]
        labels = ['happy', 'sad']
        
        self.classifier.train(features_list, labels)
        
        # Test probabilities
        features = {'pitch_mean': 150, 'energy_mean': 0.1}
        probabilities = self.classifier.get_emotion_probabilities(features)
        
        assert isinstance(probabilities, dict)
        assert len(probabilities) == len(self.classifier.emotions)
        
        # Probabilities should sum to 1.0
        total_prob = sum(probabilities.values())
        assert abs(total_prob - 1.0) < 0.001
    
    def test_save_and_load_model(self):
        """Test saving and loading model."""
        # Train the classifier
        features_list = [
            {'pitch_mean': 200, 'energy_mean': 0.2},
            {'pitch_mean': 100, 'energy_mean': 0.05}
        ]
        labels = ['happy', 'sad']
        
        self.classifier.train(features_list, labels)
        
        # Save model
        with patch('builtins.open', create=True) as mock_open:
            self.classifier.save_model('test_model.pkl')
            mock_open.assert_called_once()
        
        # Load model
        with patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = b'mock_data'
            with patch('pickle.load', return_value={
                'model': self.classifier.model,
                'scaler': self.classifier.scaler,
                'feature_names': self.classifier.feature_names,
                'emotions': self.classifier.emotions
            }):
                new_classifier = MLClassifier('test_model.pkl')
                assert new_classifier.model is not None
    
    def test_save_model_without_training(self):
        """Test saving model without training."""
        with pytest.raises(ValueError):
            self.classifier.save_model('test_model.pkl')


class TestHybridClassifier:
    """Test cases for HybridClassifier class."""
    
    def setup_method(self):
        """Setup method called before each test."""
        self.classifier = HybridClassifier()
    
    def test_initialization(self):
        """Test HybridClassifier initialization."""
        assert self.classifier.rule_classifier is not None
        assert self.classifier.ml_classifier is None
    
    def test_initialization_with_ml_model(self):
        """Test initialization with ML model path."""
        with patch('os.path.exists', return_value=False):
            classifier = HybridClassifier('test_model.pkl')
            assert classifier.ml_classifier is None
    
    def test_classify_rule_based_only(self):
        """Test classification using only rule-based approach."""
        features = {
            'pitch_mean': 200,
            'energy_mean': 0.2,
            'tempo': 150,
            'pitch_variability': 0.25
        }
        
        emotion, confidence = self.classifier.classify(features)
        
        assert emotion == 'happy'
        assert confidence > 0.5
    
    def test_classify_with_ml_model(self):
        """Test classification with ML model."""
        # Mock ML classifier
        mock_ml_classifier = Mock()
        mock_ml_classifier.classify.return_value = ('sad', 0.8)
        self.classifier.ml_classifier = mock_ml_classifier
        
        features = {
            'pitch_mean': 200,
            'energy_mean': 0.2,
            'tempo': 150,
            'pitch_variability': 0.25
        }
        
        emotion, confidence = self.classifier.classify(features)
        
        # Should use ML result due to high confidence
        assert emotion == 'sad'
        assert confidence == 0.8
    
    def test_classify_ml_failure(self):
        """Test classification when ML model fails."""
        # Mock ML classifier that raises exception
        mock_ml_classifier = Mock()
        mock_ml_classifier.classify.side_effect = Exception("ML model error")
        self.classifier.ml_classifier = mock_ml_classifier
        
        features = {
            'pitch_mean': 200,
            'energy_mean': 0.2,
            'tempo': 150,
            'pitch_variability': 0.25
        }
        
        emotion, confidence = self.classifier.classify(features)
        
        # Should fall back to rule-based
        assert emotion == 'happy'
        assert confidence > 0.5
    
    def test_get_emotion_probabilities(self):
        """Test getting emotion probabilities."""
        features = {
            'pitch_mean': 200,
            'energy_mean': 0.2,
            'tempo': 150,
            'pitch_variability': 0.25
        }
        
        probabilities = self.classifier.get_emotion_probabilities(features)
        
        assert isinstance(probabilities, dict)
        assert len(probabilities) == len(self.classifier.emotions)
        
        # Probabilities should sum to 1.0
        total_prob = sum(probabilities.values())
        assert abs(total_prob - 1.0) < 0.001
    
    def test_get_emotion_probabilities_with_ml(self):
        """Test getting probabilities with ML model."""
        # Mock ML classifier
        mock_ml_classifier = Mock()
        mock_ml_classifier.get_emotion_probabilities.return_value = {
            'happy': 0.3, 'sad': 0.7, 'angry': 0.0, 'calm': 0.0,
            'excited': 0.0, 'tired': 0.0, 'neutral': 0.0
        }
        self.classifier.ml_classifier = mock_ml_classifier
        
        features = {
            'pitch_mean': 200,
            'energy_mean': 0.2,
            'tempo': 150,
            'pitch_variability': 0.25
        }
        
        probabilities = self.classifier.get_emotion_probabilities(features)
        
        assert isinstance(probabilities, dict)
        assert len(probabilities) == len(self.classifier.emotions)
        
        # Should combine rule-based and ML probabilities
        total_prob = sum(probabilities.values())
        assert abs(total_prob - 1.0) < 0.001
    
    def test_get_emotion_probabilities_ml_failure(self):
        """Test getting probabilities when ML model fails."""
        # Mock ML classifier that raises exception
        mock_ml_classifier = Mock()
        mock_ml_classifier.get_emotion_probabilities.side_effect = Exception("ML model error")
        self.classifier.ml_classifier = mock_ml_classifier
        
        features = {
            'pitch_mean': 200,
            'energy_mean': 0.2,
            'tempo': 150,
            'pitch_variability': 0.25
        }
        
        probabilities = self.classifier.get_emotion_probabilities(features)
        
        # Should fall back to rule-based
        assert isinstance(probabilities, dict)
        assert len(probabilities) == len(self.classifier.emotions)
